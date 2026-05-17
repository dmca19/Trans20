import { createHash } from 'node:crypto'
import path from 'node:path'

import { atomicWriteFile, isNodeError, readJsonFile } from './fileUtils.js'
import { filterManualTransKeyItems } from './keyFilter.js'
import { DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE, type DefaultSourceLanguage, type DefaultTargetLanguage } from './defaultTranslationLanguages.js'
import { normalizeProjectFileId, resolveProjectFile, resolveProjectPath } from './pathUtils.js'
import { createPersistentStateStore } from './persistentStateStore.js'

export const TRANSLATION_STATE_FILE_PATH = 'translation-state.json'
export const TRANSLATED_MANUAL_TRANS_FILE_PATH = 'ManualTransFile_translated.json'

export type FrozenTranslationIndexEntry = {
    filtered_key_index: number
    raw_key_index: number
    source_key_hash: string
}

export type TranslationIndexManifest = {
    source_file_id: string
    filter_language: string
    created_at: string
    source_key_hashes: string[]
    entries: FrozenTranslationIndexEntry[]
}

export type TranslationPreflight = {
    source_language: DefaultSourceLanguage
    target_language: DefaultTargetLanguage
    domain_hint: string
    text_profile: string[]
    style_guidance: string[]
    format_protection: string[]
    glossary_usage: string[]
    context_usage: string[]
    quality_cautions: string[]
    submitted_at: string
}

export type TranslationRecord = {
    filtered_key_index: number
    source_key_hash: string
    translation_value: string
    submitted_at: string
    submitted_by?: string
    batch_id?: string
}

export type TranslationState = {
    version: 1
    index_manifest: TranslationIndexManifest|null
    preflight: TranslationPreflight|null
    translations: TranslationRecord[]
    meta: {
        created_at: string
        updated_at: string
    }
}

export type ManualTransData = {
    relativePath: string
    keys: string[]
    values: unknown[]
}

export type TranslationExportResult = {
    outputPath: string
    translatedCount: number
    preservedCount: number
    totalKeys: number
}

const translationStateStore = createPersistentStateStore<TranslationState>({
    resolveFilePath: resolveTranslationStateFilePath,
    loadFromDisk: readTranslationStateFromDisk,
})

export async function loadTranslationState (root: string): Promise<TranslationState> {
    return readTranslationState(root, state => structuredClone(state) as TranslationState)
}

export async function readTranslationState<T> (
    root: string,
    reader: (state: TranslationState) => Promise<T>|T,
): Promise<T> {
    return translationStateStore.read(root, reader)
}

export async function updateTranslationState<T> (
    root: string,
    updater: (state: TranslationState) => Promise<T>|T,
): Promise<T> {
    return translationStateStore.update(root, updater)
}

export async function saveTranslationState (root: string, state: TranslationState): Promise<void> {
    await translationStateStore.save(root, state)
}

export async function flushTranslationState (root: string): Promise<void> {
    await translationStateStore.flush(root)
}

export async function flushAllTranslationStores (): Promise<void> {
    await translationStateStore.flushAll()
}

export async function closeTranslationStore (root: string): Promise<void> {
    await translationStateStore.close(root)
}

// Same ManualTrans object contract as glossaryTools.loadManualTransData.
export async function loadManualTransData (root: string, manualTransFile: string): Promise<ManualTransData> {
    const file = await resolveProjectFile(root, manualTransFile, {
        realPathMessage: `Path resolves outside the run directory: ${manualTransFile}`,
    })
    const parsed = await readJsonFile(file.realFilePath, file.relativePath)

    if (!isRecord(parsed)) {
        throw new Error(`${file.relativePath} must be a JSON object.`)
    }

    const keys = Object.keys(parsed)
    return {
        relativePath: file.relativePath,
        keys,
        values: keys.map(key => parsed[key]),
    }
}

export async function exportTranslatedManualTransFile (
    root: string,
    manualTransFile: string,
    outputRelativePath = TRANSLATED_MANUAL_TRANS_FILE_PATH,
    stateRoot = root,
): Promise<TranslationExportResult> {
    const { manifest, manualTrans } = await ensureTranslationIndexManifest(root, manualTransFile, DEFAULT_SOURCE_LANGUAGE, stateRoot)
    const state = await readTranslationState(stateRoot, item => item)
    const translationByFilteredIndex = new Map(state.translations.map(item => [item.filtered_key_index, item]))
    const manifestByRawIndex = new Map(manifest.entries.map(entry => [entry.raw_key_index, entry]))
    const translatedRawIndexes = new Set<number>()
    const output: Record<string, unknown> = {}

    for (const entry of manifest.entries) {
        const translation = translationByFilteredIndex.get(entry.filtered_key_index)

        if (translation) {
            translatedRawIndexes.add(entry.raw_key_index)
        }
    }

    for (const [rawIndex, key] of manualTrans.keys.entries()) {
        const manifestEntry = manifestByRawIndex.get(rawIndex)
        const translation = manifestEntry ? translationByFilteredIndex.get(manifestEntry.filtered_key_index) : null

        output[key] = translation?.translation_value ?? manualTrans.values[rawIndex]
    }

    const outputPath = path.resolve(stateRoot, outputRelativePath)
    const projectPath = resolveProjectPath(root, outputPath, {
        rootPathMessage: `Translation export path is outside the project root: ${outputRelativePath}`,
        rejectRootPathMessage: `Translation export path is outside the project root: ${outputRelativePath}`,
    })

    await atomicWriteFile(outputPath, `${JSON.stringify(output, null, 2)}\n`)

    return {
        outputPath: projectPath.normalizedRelativePath,
        translatedCount: translatedRawIndexes.size,
        preservedCount: manualTrans.keys.length - translatedRawIndexes.size,
        totalKeys: manualTrans.keys.length,
    }
}

export async function ensureTranslationIndexManifest (
    root: string,
    manualTransFile: string,
    filterLanguage: string,
    stateRoot = root,
): Promise<{ manifest: TranslationIndexManifest, manualTrans: ManualTransData }> {
    const manualTrans = await loadManualTransData(root, manualTransFile)

    return updateTranslationState(stateRoot, async state => {
        if (!state.index_manifest) {
            const keyItems = await filterManualTransKeyItems(root, manualTrans.keys, filterLanguage)
            const now = new Date().toISOString()
            state.index_manifest = {
                source_file_id: manualTrans.relativePath,
                filter_language: filterLanguage,
                created_at: now,
                source_key_hashes: manualTrans.keys.map(sha256SourceText),
                entries: keyItems.map(item => ({
                    filtered_key_index: item.filteredIndex,
                    raw_key_index: item.originalIndex,
                    source_key_hash: sha256SourceText(manualTrans.keys[item.originalIndex] ?? ''),
                })),
            }
            state.meta.updated_at = now
        }

        if (state.index_manifest.filter_language !== filterLanguage) {
            throw new Error(`Frozen translation index filter language is ${state.index_manifest.filter_language}, not ${filterLanguage}.`)
        }

        validateManifestAgainstSource(state.index_manifest, manualTrans)
        return {
            manifest: structuredClone(state.index_manifest) as TranslationIndexManifest,
            manualTrans,
        }
    })
}

// Same stored source-hash contract as glossaryTools: sha256: plus hex.
export function sha256SourceText (value: string): string {
    return `sha256:${createHash('sha256').update(value).digest('hex')}`
}

function validateManifestAgainstSource (manifest: TranslationIndexManifest, manualTrans: ManualTransData): void {
    if (manifest.source_key_hashes.length > 0) {
        if (manifest.source_key_hashes.length !== manualTrans.keys.length) {
            throw new Error(`Frozen translation index source key count changed from ${manifest.source_key_hashes.length} to ${manualTrans.keys.length}.`)
        }

        for (const [index, expectedHash] of manifest.source_key_hashes.entries()) {
            if (sha256SourceText(manualTrans.keys[index] ?? '') !== expectedHash) {
                throw new Error(`Frozen translation index source key changed at raw_key_index ${index}.`)
            }
        }
    } else if (normalizeProjectFileId(manifest.source_file_id) !== normalizeProjectFileId(manualTrans.relativePath)) {
        throw new Error(`Frozen translation index source ${manifest.source_file_id} does not match current source ${manualTrans.relativePath}.`)
    }

    for (const entry of manifest.entries) {
        const sourceKey = manualTrans.keys[entry.raw_key_index]

        if (typeof sourceKey !== 'string' || sha256SourceText(sourceKey) !== entry.source_key_hash) {
            throw new Error(`Frozen translation index is stale at filtered_key_index ${entry.filtered_key_index}.`)
        }
    }
}

async function readTranslationStateFromDisk (filePath: string): Promise<TranslationState> {
    try {
        const parsedState = await readJsonFile(filePath, TRANSLATION_STATE_FILE_PATH)
        return parseTranslationState(parsedState)
    } catch (error) {
        if (isNodeError(error) && error.code === 'ENOENT') {
            return createEmptyTranslationState()
        }

        throw error
    }
}

function createEmptyTranslationState (): TranslationState {
    const now = new Date().toISOString()

    return {
        version: 1,
        index_manifest: null,
        preflight: null,
        translations: [],
        meta: {
            created_at: now,
            updated_at: now,
        },
    }
}

function parseTranslationState (value: unknown): TranslationState {
    if (!isRecord(value)) {
        throw new Error(`${TRANSLATION_STATE_FILE_PATH} must contain a JSON object.`)
    }

    const fallback = createEmptyTranslationState()

    return {
        version: 1,
        index_manifest: isTranslationIndexManifest(value.index_manifest)
            ? {
                ...value.index_manifest,
                source_key_hashes: value.index_manifest.source_key_hashes ?? [],
            }
            : null,
        preflight: isTranslationPreflight(value.preflight) ? value.preflight : null,
        translations: Array.isArray(value.translations) ? value.translations.filter(isTranslationRecord) : [],
        meta: {
            created_at: readMetaTimestamp(value, 'created_at') ?? fallback.meta.created_at,
            updated_at: readMetaTimestamp(value, 'updated_at') ?? fallback.meta.updated_at,
        },
    }
}

function resolveTranslationStateFilePath (root: string): string {
    return path.join(root, TRANSLATION_STATE_FILE_PATH)
}

function isTranslationIndexManifest (value: unknown): value is TranslationIndexManifest {
    return isRecord(value)
        && typeof value.source_file_id === 'string'
        && typeof value.filter_language === 'string'
        && typeof value.created_at === 'string'
        && (value.source_key_hashes === undefined || stringArray(value.source_key_hashes))
        && Array.isArray(value.entries)
        && value.entries.every(isFrozenTranslationIndexEntry)
}

function isFrozenTranslationIndexEntry (value: unknown): value is FrozenTranslationIndexEntry {
    return isRecord(value)
        && Number.isInteger(value.filtered_key_index)
        && Number.isInteger(value.raw_key_index)
        && typeof value.source_key_hash === 'string'
}

function isTranslationPreflight (value: unknown): value is TranslationPreflight {
    return isRecord(value)
        && value.source_language === DEFAULT_SOURCE_LANGUAGE
        && value.target_language === DEFAULT_TARGET_LANGUAGE
        && typeof value.domain_hint === 'string'
        && typeof value.submitted_at === 'string'
        && stringArray(value.text_profile)
        && stringArray(value.style_guidance)
        && stringArray(value.format_protection)
        && stringArray(value.glossary_usage)
        && stringArray(value.context_usage)
        && stringArray(value.quality_cautions)
}

function isTranslationRecord (value: unknown): value is TranslationRecord {
    return isRecord(value)
        && Number.isInteger(value.filtered_key_index)
        && typeof value.source_key_hash === 'string'
        && typeof value.translation_value === 'string'
        && typeof value.submitted_at === 'string'
}

function stringArray (value: unknown): value is string[] {
    return Array.isArray(value) && value.every(item => typeof item === 'string')
}

function readMetaTimestamp (value: Record<string, unknown>, key: string): string|null {
    if (!isRecord(value.meta)) {
        return null
    }

    return typeof value.meta[key] === 'string' ? value.meta[key] : null
}

// Same plain-object guard as agent/taskStore/glossaryStore/glossaryTools isRecord.
function isRecord (value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
}
