import { createHash } from 'node:crypto'

import { tool, type StructuredToolInterface } from '@langchain/core/tools'
import { distance as levenshteinDistance } from 'fastest-levenshtein'
// LangChain tool schemas intentionally use Zod v3; do not switch this to the package default v4 import.
import { z } from 'zod/v3'

import { collectCharacterContextWarnings } from './characterContextWarnings.js'
import { createYieldController } from './eventLoop.js'
import {
    appliesToValues,
    type EntryStatus,
    type EntryType,
    entryStatusValues,
    entryTypeValues,
    type GlossaryEntry,
    type GlossaryEvidence,
    type GlossaryState,
    type GlossaryTerm,
    policyStrengthValues,
    readGlossaryState,
    termStatusValues,
    termTypeValues,
} from './glossaryStore.js'
import {
    formatTermForAgent,
    getAliasTexts,
    getSourceVariants,
    selectorsToAgentApplicabilityFields,
} from './glossaryAliasUtils.js'
import {
    ensureTranslationIndexManifest,
    type FrozenTranslationIndexEntry,
    type ManualTransData,
    readTranslationState,
    type TranslationIndexManifest,
    type TranslationPreflight,
    type TranslationRecord,
    updateTranslationState,
} from './translationStore.js'
import { normalizeProjectFileId } from './pathUtils.js'
import { DEFAULT_FILTER_LANGUAGE, DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE } from './defaultTranslationLanguages.js'
import { jsonOutput, truncate, withToolLogging, type ToolCallLogger } from './toolRuntime.js'

const DEFAULT_CONTEXT_WINDOW = 50
const MAX_CONTEXT_WINDOW = 300
const DEFAULT_EVIDENCE_CONTEXT_WINDOW = 20
const MAX_EVIDENCE_CONTEXT_WINDOW = 500
const DEFAULT_MAX_CHARS_PER_KEY = 500
const MAX_CHARS_PER_KEY = 2000
const DEFAULT_ENTRY_LIMIT_PER_TERM = 10
const MAX_ENTRY_LIMIT_PER_TERM = 50
const MAX_LOOKUP_TERMS = 100
const MAX_BATCH_TRANSLATION_ITEMS = 100
const DEFAULT_TRANSLATION_MEMORY_LIMIT = 20
const MAX_TRANSLATION_MEMORY_LIMIT = 100
const DEFAULT_TRANSLATION_MEMORY_MAX_CHARS_PER_FIELD = 300
const MAX_TRANSLATION_MEMORY_MAX_CHARS_PER_FIELD = 1000

const matchModeValues = ['exact', 'alias', 'compound', 'fuzzy', 'regex'] as const
const translationMemoryFieldValues = ['source_key', 'translation_value'] as const

export type CreateTranslationToolsOptions = {
    manualTransFile: string
    translationStateRoot?: string
    glossaryStateRoot?: string
    filterLanguage?: string
    enableTranslationMemorySearch?: boolean
    enableCharacterGenderWarnings?: boolean
    batchStartIndex?: number
    batchEndIndex?: number
    batchId?: string
    submittedBy?: string
    onPreflightSubmitted?: (preflight: TranslationPreflight) => void
    onTranslationBatchSubmitted?: (submission: TranslationBatchSubmission) => void
}

export type TranslationBatchSubmission = {
    submitted_count: number
    filtered_key_indexes: number[]
}

type TranslationToolSession = {
    returnedTermIds: Set<string>
    returnedEvidenceIds: Set<string>
}

type TranslationRuntime = {
    manifest: TranslationIndexManifest
    manualTrans: ManualTransData
    manifestByFiltered: Map<number, FrozenTranslationIndexEntry>
    filteredByRaw: Map<number, number>
}

type LookupMatch = {
    term: GlossaryTerm
    entries: GlossaryEntry[]
    evidence: GlossaryEvidence[]
    matchReason: string
    matchedFilteredKeyIndexes: number[]
    matchedSourceIndexes: MatchedSourceIndex[]
    score: number
    distance: number
    translationRules: GlossaryEntry[]
    applicableTranslationRules: GlossaryEntry[]
    otherTranslationRules: GlossaryEntry[]
}

type MatchedSourceIndex = {
    index: number
    text: string
    matched_filtered_key_indexes: number[]
}

type EvidenceSummary = {
    evidence_id: string
    term_id: string
    entry_id: string
    filtered_key_index: number
    quote: string
}

type TranslationEntryView = Record<string, unknown> & {
    evidence_ids?: string[]
    evidence_summary?: EvidenceSummary[]
}

type TranslationTermEntriesResult = {
    terms: Array<{
        term_id: string
        source_text: string|null
        entries: TranslationEntryView[]
    }>
    next_cursor: string|null
    warnings: TranslationToolWarning[]
}

type TranslationToolWarning = {
    code: string
    entry_id?: string
    field: string
    message: string
}

type TranslationGlossaryTermMatch = {
    term: Record<string, unknown> & { term_id: string }
    match: Record<string, unknown>
    entries: Record<string, unknown>[]
    evidence: EvidenceSummary[]
}

type TranslationGlossaryEntryMatch = {
    entry: Record<string, unknown>
    term?: Record<string, unknown> & { term_id: string }
    evidence: EvidenceSummary[]
}

const nonEmptyString = z.string().trim().min(1)

const submitTranslationPreflightSchema = z.object({
    source_language: z.literal(DEFAULT_SOURCE_LANGUAGE),
    target_language: z.literal(DEFAULT_TARGET_LANGUAGE),
    domain_hint: z.enum(['game_script', 'game_dialogue', 'game_ui', 'mixed_game_text', 'mixed_or_unknown']),
    text_profile: z.array(nonEmptyString).min(3).max(8),
    style_guidance: z.array(nonEmptyString).min(3).max(8),
    format_protection: z.array(nonEmptyString).min(3).max(8),
    glossary_usage: z.array(nonEmptyString).min(3).max(8),
    context_usage: z.array(nonEmptyString).min(3).max(8),
    quality_cautions: z.array(nonEmptyString).min(3).max(8),
}).strict()

const getTranslationBatchSchema = z.object({}).strict()

const lookupTranslationTermsSchema = z.object({
    speaker: z.string().trim().optional(),
    include_translation_rules: z.boolean().optional().default(false),
    limit: z.number().int().min(1).max(MAX_LOOKUP_TERMS).optional().default(MAX_LOOKUP_TERMS),
}).strict()

const getTranslationTermEntriesSchema = z.object({
    term_ids: z.array(nonEmptyString).nonempty(),
    entry_types: z.array(z.enum(entryTypeValues)).optional(),
    limit_per_term: z.number().int().min(1).max(MAX_ENTRY_LIMIT_PER_TERM).optional().default(DEFAULT_ENTRY_LIMIT_PER_TERM),
    cursor: z.string().trim().optional(),
    include_evidence_summary: z.enum(['ids_only', 'summary', 'none']).optional().default('ids_only'),
}).strict()

const getTranslationTermContextSchema = z.object({
    evidence_ids: z.array(nonEmptyString).nonempty(),
    window_before: z.number().int().min(0).max(MAX_EVIDENCE_CONTEXT_WINDOW).optional().default(DEFAULT_EVIDENCE_CONTEXT_WINDOW),
    window_after: z.number().int().min(0).max(MAX_EVIDENCE_CONTEXT_WINDOW).optional().default(DEFAULT_EVIDENCE_CONTEXT_WINDOW),
    max_chars_per_key: z.number().int().min(1).max(MAX_CHARS_PER_KEY).optional().default(DEFAULT_MAX_CHARS_PER_KEY),
}).strict()

const submitTranslationBatchSchema = z.object({
    translations: z.array(z.object({
        filtered_key_index: z.number().int().min(0),
        translation_value: z.string(),
    }).strict()).nonempty().max(MAX_BATCH_TRANSLATION_ITEMS),
}).strict()

const getTranslationContextSchema = z.object({
    before: z.number().int().min(0).max(MAX_CONTEXT_WINDOW).optional().default(DEFAULT_CONTEXT_WINDOW),
    after: z.number().int().min(0).max(MAX_CONTEXT_WINDOW).optional().default(DEFAULT_CONTEXT_WINDOW),
}).strict()

const searchTranslationMemorySchema = z.object({
    query: nonEmptyString,
    fields: z.array(z.enum(translationMemoryFieldValues)).nonempty().optional().default(['translation_value']),
    is_regex: z.boolean().optional().default(false),
    case_sensitive: z.boolean().optional().default(false),
    include_unsubmitted_source_matches: z.boolean().optional().default(false),
    limit: z.number().int().min(1).max(MAX_TRANSLATION_MEMORY_LIMIT).optional().default(DEFAULT_TRANSLATION_MEMORY_LIMIT),
    cursor: z.string().trim().optional(),
    max_chars_per_field: z.number().int().min(20).max(MAX_TRANSLATION_MEMORY_MAX_CHARS_PER_FIELD).optional().default(DEFAULT_TRANSLATION_MEMORY_MAX_CHARS_PER_FIELD),
}).strict()

// Same glossary query shape, but read-only with translation-local limits/defaults.
const queryGlossaryTermsSchema = z.object({
    queries: z.array(z.object({
        text: nonEmptyString,
        match_modes: z.array(z.enum(matchModeValues)).nonempty(),
        case_sensitive: z.boolean().optional().default(false),
    }).strict()).nonempty(),
    source_language: nonEmptyString,
    include_entries: z.boolean().optional().default(false),
    include_evidence: z.boolean().optional().default(false),
    entry_statuses: z.array(z.enum(entryStatusValues)).nullable().optional(),
    term_statuses: z.array(z.enum(termStatusValues)).nullable().optional(),
    limit: z.number().int().min(1).max(MAX_LOOKUP_TERMS),
}).strict()

const searchGlossaryEntriesSchema = z.object({
    entry_filter: z.object({
        entry_types: z.array(z.enum(entryTypeValues)).nullable().optional(),
        entry_statuses: z.array(z.enum(entryStatusValues)).nullable().optional(),
        applies_to: z.array(z.enum(appliesToValues)).nullable().optional(),
        target_language: z.string().trim().min(1).nullable().optional(),
        policy_strength: z.array(z.enum(policyStrengthValues)).nullable().optional(),
        domain: z.string().trim().min(1).nullable().optional(),
        text: z.string().trim().min(1).nullable().optional(),
        text_fields: z.array(z.enum([
            'content.summary',
            'content.description',
            'target.preferred_translation',
            'target.alternative_translations',
            'target.forbidden_translations',
            'policy.notes',
            'applicability.applies_when',
            'applicability.does_not_apply_when',
        ])).nullable().optional(),
        is_regex: z.boolean().optional().default(false),
        case_sensitive: z.boolean().optional().default(false),
    }).strict().nullable().optional(),
    term_filter: z.object({
        source_language: z.string().trim().min(1).nullable().optional(),
        term_types: z.array(z.enum(termTypeValues)).nullable().optional(),
        term_statuses: z.array(z.enum(termStatusValues)).nullable().optional(),
    }).strict().nullable().optional(),
    include_term: z.boolean().optional().default(false),
    include_evidence: z.boolean().optional().default(false),
    limit: z.number().int().min(1).max(MAX_LOOKUP_TERMS),
}).strict()

const evidenceContextSchema = z.object({
    evidence_ids: z.array(nonEmptyString).nonempty(),
    window_before: z.number().int().min(0).max(MAX_EVIDENCE_CONTEXT_WINDOW).optional().default(DEFAULT_EVIDENCE_CONTEXT_WINDOW),
    window_after: z.number().int().min(0).max(MAX_EVIDENCE_CONTEXT_WINDOW).optional().default(DEFAULT_EVIDENCE_CONTEXT_WINDOW),
    max_chars_per_key: z.number().int().min(1).max(MAX_CHARS_PER_KEY).optional().default(DEFAULT_MAX_CHARS_PER_KEY),
}).strict()

export function createTranslationPreflightTools (
    root: string,
    onToolEvent: ToolCallLogger,
    options: CreateTranslationToolsOptions,
): StructuredToolInterface[] {
    const manualTransFile = options.manualTransFile
    const translationStateRoot = options.translationStateRoot ?? root
    const filterLanguage = options.filterLanguage ?? DEFAULT_FILTER_LANGUAGE

    return [
        tool(async input => withToolLogging('submit_translation_preflight', input, onToolEvent, async () => {
            const validation = submitTranslationPreflightSchema.safeParse(input)

            if (!validation.success) {
                return jsonOutput(validationErrorOutput(validation.error.errors.map(error => ({
                    field: error.path.join('.') || '(root)',
                    message: error.message,
                }))))
            }

            // Preflight text regex screening is currently disabled; keep validatePreflightText below for a possible future re-enable.

            await ensureTranslationIndexManifest(root, manualTransFile, filterLanguage, translationStateRoot)
            const now = new Date().toISOString()
            const preflight: TranslationPreflight = {
                ...validation.data,
                submitted_at: now,
            }
            await updateTranslationState(translationStateRoot, state => {
                state.preflight = preflight
                state.meta.updated_at = now
            })
            options.onPreflightSubmitted?.(preflight)

            return jsonOutput({
                ok: true,
                retry_required: false,
                submitted_at: now,
            })
        }), {
            name: 'submit_translation_preflight',
            description: 'Submit project-level translation guidance. Does not translate keys or modify glossary records.',
            schema: submitTranslationPreflightSchema,
        }),
    ]
}

export function createTranslationWorkerTools (
    root: string,
    onToolEvent: ToolCallLogger,
    options: CreateTranslationToolsOptions,
): StructuredToolInterface[] {
    if (!Number.isInteger(options.batchStartIndex) || !Number.isInteger(options.batchEndIndex)) {
        throw new Error('createTranslationWorkerTools requires batchStartIndex and batchEndIndex.')
    }

    const manualTransFile = options.manualTransFile
    const translationStateRoot = options.translationStateRoot ?? root
    const glossaryStateRoot = options.glossaryStateRoot ?? root
    const filterLanguage = options.filterLanguage ?? DEFAULT_FILTER_LANGUAGE
    const enableTranslationMemorySearch = options.enableTranslationMemorySearch ?? true
    const enableCharacterGenderWarnings = options.enableCharacterGenderWarnings ?? false
    const batchStartIndex = options.batchStartIndex
    const batchEndIndex = options.batchEndIndex

    if (batchStartIndex === undefined || batchEndIndex === undefined) {
        throw new Error('createTranslationWorkerTools requires batchStartIndex and batchEndIndex.')
    }

    if (batchStartIndex > batchEndIndex) {
        throw new Error('createTranslationWorkerTools requires batchStartIndex <= batchEndIndex.')
    }

    const session: TranslationToolSession = {
        returnedTermIds: new Set<string>(),
        returnedEvidenceIds: new Set<string>(),
    }

    return [
        tool(async input => withToolLogging('get_translation_batch', input, onToolEvent, async () => {
            const validation = getTranslationBatchSchema.safeParse(input)

            if (!validation.success) {
                return jsonOutput(zodErrorOutput(validation.error))
            }

            const runtime = await loadRuntime(root, translationStateRoot, manualTransFile, filterLanguage)
            const translations = await readTranslationState(translationStateRoot, state => state.translations)
            const submitted = new Map(translations.map(item => [item.filtered_key_index, item]))
            const items = getBatchManifestEntries(runtime.manifest, batchStartIndex, batchEndIndex).map(entry => {
                const sourceKey = runtime.manualTrans.keys[entry.raw_key_index] ?? ''
                return {
                    filtered_key_index: entry.filtered_key_index,
                    source_key: sourceKey,
                    current_value: runtime.manualTrans.values[entry.raw_key_index] ?? null,
                    protection_risks: analyzeProtectionRisks(sourceKey),
                    translation_status: submitted.has(entry.filtered_key_index) ? 'translated' : 'untranslated',
                    is_current_batch: true,
                }
            })

            return jsonOutput({
                ok: true,
                batch_start_index: batchStartIndex,
                batch_end_index: batchEndIndex,
                items,
                warnings: [],
            })
        }), {
            name: 'get_translation_batch',
            description: 'Return the current translation batch using frozen filtered_key_index values only.',
            schema: getTranslationBatchSchema,
        }),
        tool(async input => withToolLogging('lookup_translation_terms', input, onToolEvent, async () => {
            const validation = lookupTranslationTermsSchema.safeParse(input)

            if (!validation.success) {
                return jsonOutput(zodErrorOutput(validation.error))
            }

            const runtime = await loadRuntime(root, translationStateRoot, manualTransFile, filterLanguage)
            const glossary = await readGlossaryState(glossaryStateRoot, state => state)
            const matches = findTranslationTermMatches(glossary, runtime, batchStartIndex, batchEndIndex, validation.data.speaker ?? '')
                .slice(0, validation.data.limit)

            for (const match of matches) {
                session.returnedTermIds.add(match.term.term_id)
            }

            return jsonOutput({
                ok: true,
                matched_terms: matches.map(match => formatLookupMatch(match, validation.data.include_translation_rules, enableCharacterGenderWarnings)),
                warnings: enableCharacterGenderWarnings ? collectLookupCharacterContextWarnings(matches) : [],
            })
        }), {
            name: 'lookup_translation_terms',
            description: 'Lookup active approved glossary terms relevant to the current batch using filtered indexes.',
            schema: lookupTranslationTermsSchema,
        }),
        tool(async input => withToolLogging('get_translation_term_entries', input, onToolEvent, async () => {
            const validation = getTranslationTermEntriesSchema.safeParse(input)

            if (!validation.success) {
                return jsonOutput(zodErrorOutput(validation.error))
            }

            const unqueriedTermIds = validation.data.term_ids.filter(termId => !session.returnedTermIds.has(termId))

            if (unqueriedTermIds.length > 0) {
                return jsonOutput(validationErrorOutput(unqueriedTermIds.map(termId => ({
                    field: 'term_ids',
                    message: `${termId} was not returned by lookup_translation_terms in this session.`,
                }))))
            }

            const runtime = await loadRuntime(root, translationStateRoot, manualTransFile, filterLanguage)
            const glossary = await readGlossaryState(glossaryStateRoot, state => state)
            const output = getTranslationTermEntries(glossary, runtime, validation.data)

            for (const termResult of output.terms) {
                for (const entry of termResult.entries) {
                    for (const evidenceId of entry.evidence_ids ?? []) {
                        session.returnedEvidenceIds.add(evidenceId)
                    }
                    for (const evidence of entry.evidence_summary ?? []) {
                        session.returnedEvidenceIds.add(evidence.evidence_id)
                    }
                }
            }

            return jsonOutput({
                ok: true,
                ...output,
            })
        }), {
            name: 'get_translation_term_entries',
            description: 'Return approved entries for terms previously returned by lookup_translation_terms.',
            schema: getTranslationTermEntriesSchema,
        }),
        tool(async input => withToolLogging('get_translation_term_context', input, onToolEvent, async () => {
            const validation = getTranslationTermContextSchema.safeParse(input)

            if (!validation.success) {
                return jsonOutput(zodErrorOutput(validation.error))
            }

            const invalidEvidenceIds = validation.data.evidence_ids.filter(evidenceId => !session.returnedEvidenceIds.has(evidenceId))

            if (invalidEvidenceIds.length > 0) {
                return jsonOutput(validationErrorOutput(invalidEvidenceIds.map(evidenceId => ({
                    field: 'evidence_ids',
                    message: `${evidenceId} was not returned by lookup_translation_terms or get_translation_term_entries in this session.`,
                }))))
            }

            const runtime = await loadRuntime(root, translationStateRoot, manualTransFile, filterLanguage)
            const glossary = await readGlossaryState(glossaryStateRoot, state => state)

            return jsonOutput({
                ok: true,
                contexts: createEvidenceContexts(glossary.evidence, runtime, validation.data.evidence_ids, {
                    before: validation.data.window_before,
                    after: validation.data.window_after,
                    maxCharsPerKey: validation.data.max_chars_per_key,
                    batchStartIndex,
                    batchEndIndex,
                }),
                warnings: [],
            })
        }), {
            name: 'get_translation_term_context',
            description: 'Return filtered-index evidence context for evidence ids exposed by translation glossary tools.',
            schema: getTranslationTermContextSchema,
        }),
        tool(async input => withToolLogging('submit_translation_batch', input, onToolEvent, async () => {
            const validation = submitTranslationBatchSchema.safeParse(input)

            if (!validation.success) {
                return jsonOutput(zodErrorOutput(validation.error))
            }

            const runtime = await loadRuntime(root, translationStateRoot, manualTransFile, filterLanguage)
            const batchEntries = getBatchManifestEntries(runtime.manifest, batchStartIndex, batchEndIndex)
            const result = await submitTranslationBatch(translationStateRoot, runtime, batchEntries, validation.data.translations, {
                batchId: options.batchId,
                submittedBy: options.submittedBy,
            })

            if (isTranslationBatchSubmissionResult(result)) {
                options.onTranslationBatchSubmitted?.({
                    submitted_count: result.submitted_count,
                    filtered_key_indexes: result.filtered_key_indexes,
                })
            }

            return jsonOutput(result)
        }), {
            name: 'submit_translation_batch',
            description: 'Submit translations for every key in the current batch using filtered_key_index only.',
            schema: submitTranslationBatchSchema,
        }),
        tool(async input => withToolLogging('get_translation_context', input, onToolEvent, async () => {
            const validation = getTranslationContextSchema.safeParse(input)

            if (!validation.success) {
                return jsonOutput(zodErrorOutput(validation.error))
            }

            const runtime = await loadRuntime(root, translationStateRoot, manualTransFile, filterLanguage)
            const translations = await readTranslationState(translationStateRoot, state => state.translations)
            const startIndex = Math.max(0, batchStartIndex - validation.data.before)
            const endIndex = batchEndIndex + validation.data.after
            const translationByFiltered = new Map(translations.map(item => [item.filtered_key_index, item]))
            const items = runtime.manifest.entries
                .filter(entry => entry.filtered_key_index >= startIndex && entry.filtered_key_index <= endIndex)
                .map(entry => {
                    const translation = translationByFiltered.get(entry.filtered_key_index)
                    return {
                        filtered_key_index: entry.filtered_key_index,
                        source_key: runtime.manualTrans.keys[entry.raw_key_index] ?? '',
                        translation_value: translation?.translation_value ?? null,
                        distance_from_current_batch: distanceFromBatch(entry.filtered_key_index, batchStartIndex, batchEndIndex),
                    }
                })

            return jsonOutput({
                ok: true,
                items,
                warnings: [],
            })
        }), {
            name: 'get_translation_context',
            description: 'Return nearby source keys; translation_value is the submitted translation when available, otherwise null.',
            schema: getTranslationContextSchema,
        }),
        ...(enableTranslationMemorySearch
            ? [
                tool(async input => withToolLogging('search_translation_memory', input, onToolEvent, async () => {
                    const validation = searchTranslationMemorySchema.safeParse(input)

                    if (!validation.success) {
                        return jsonOutput(zodErrorOutput(validation.error))
                    }

                    if (validation.data.is_regex) {
                        const regexErrors = validateRegexText('query', validation.data.query)
                        if (regexErrors.length > 0) {
                            return jsonOutput(validationErrorOutput(regexErrors))
                        }
                    }

                    const runtime = await loadRuntime(root, translationStateRoot, manualTransFile, filterLanguage)
                    const translations = await readTranslationState(translationStateRoot, state => state.translations)
                    const output = searchTranslationMemory(runtime, translations, validation.data, {
                        batchStartIndex,
                        batchEndIndex,
                    })

                    return jsonOutput({
                        ok: true,
                        ...output,
                    })
                }), {
                    name: 'search_translation_memory',
                    description: 'Search submitted translation memory across all filtered keys. Defaults to submitted translation_value matches sorted by relevance. To search original source text, explicitly pass fields: ["source_key"]. Source_key matches without a submitted translation are returned only when include_unsubmitted_source_matches is true.',
                    schema: searchTranslationMemorySchema,
                }),
            ]
            : []),
        ...createTranslationGlossaryTools(root, onToolEvent, {
            manualTransFile,
            translationStateRoot,
            glossaryStateRoot,
            filterLanguage,
            batchStartIndex,
            batchEndIndex,
            session,
        }),
    ]
}

function createTranslationGlossaryTools (
    root: string,
    onToolEvent: ToolCallLogger,
    options: {
        manualTransFile: string
        translationStateRoot: string
        glossaryStateRoot: string
        filterLanguage: string
        batchStartIndex: number
        batchEndIndex: number
        session: TranslationToolSession
    },
): StructuredToolInterface[] {
    return [
        tool(async input => withToolLogging('query_glossary_terms', input, onToolEvent, async () => {
            const validation = queryGlossaryTermsSchema.safeParse(input)

            if (!validation.success) {
                return jsonOutput(zodErrorOutput(validation.error))
            }

            const regexErrors = validateQueryRegexes(validation.data.queries)

            if (regexErrors.length > 0) {
                return jsonOutput(validationErrorOutput(regexErrors))
            }

            const runtime = await loadRuntime(root, options.translationStateRoot, options.manualTransFile, options.filterLanguage)
            const glossary = await readGlossaryState(options.glossaryStateRoot, state => state)
            const output = await queryTranslationGlossaryTerms(glossary, runtime, validation.data)

            for (const match of output.matched_terms) {
                options.session.returnedTermIds.add(match.term.term_id)
                for (const evidence of match.evidence) {
                    options.session.returnedEvidenceIds.add(evidence.evidence_id)
                }
            }

            return jsonOutput({
                ok: true,
                matched_terms: output.matched_terms,
                warnings: output.warnings,
            })
        }), {
            name: 'query_glossary_terms',
            description: 'Translation-safe glossary term query. Returns active terms and approved entries only.',
            schema: queryGlossaryTermsSchema,
        }),
        tool(async input => withToolLogging('search_glossary_entries', input, onToolEvent, async () => {
            const validation = searchGlossaryEntriesSchema.safeParse(input)

            if (!validation.success) {
                return jsonOutput(zodErrorOutput(validation.error))
            }

            const regexErrors = validateSearchRegex(validation.data.entry_filter ?? null)

            if (regexErrors.length > 0) {
                return jsonOutput(validationErrorOutput(regexErrors))
            }

            const runtime = await loadRuntime(root, options.translationStateRoot, options.manualTransFile, options.filterLanguage)
            const glossary = await readGlossaryState(options.glossaryStateRoot, state => state)
            const output = await searchTranslationGlossaryEntries(glossary, runtime, validation.data)

            for (const match of output.matched_entries) {
                if (match.term) {
                    options.session.returnedTermIds.add(match.term.term_id)
                }
                for (const evidence of match.evidence) {
                    options.session.returnedEvidenceIds.add(evidence.evidence_id)
                }
            }

            return jsonOutput({
                ok: true,
                matched_entries: output.matched_entries,
                warnings: output.warnings,
            })
        }), {
            name: 'search_glossary_entries',
            description: 'Translation-safe glossary entry search. Returns approved entries only.',
            schema: searchGlossaryEntriesSchema,
        }),
        tool(async input => withToolLogging('get_evidence_context', input, onToolEvent, async () => {
            const validation = evidenceContextSchema.safeParse(input)

            if (!validation.success) {
                return jsonOutput(zodErrorOutput(validation.error))
            }

            const invalidEvidenceIds = validation.data.evidence_ids.filter(evidenceId => !options.session.returnedEvidenceIds.has(evidenceId))

            if (invalidEvidenceIds.length > 0) {
                return jsonOutput(validationErrorOutput(invalidEvidenceIds.map(evidenceId => ({
                    field: 'evidence_ids',
                    message: `${evidenceId} was not returned by a translation glossary query in this session.`,
                }))))
            }

            const runtime = await loadRuntime(root, options.translationStateRoot, options.manualTransFile, options.filterLanguage)
            const glossary = await readGlossaryState(options.glossaryStateRoot, state => state)

            return jsonOutput({
                ok: true,
                contexts: createEvidenceContexts(glossary.evidence, runtime, validation.data.evidence_ids, {
                    before: validation.data.window_before,
                    after: validation.data.window_after,
                    maxCharsPerKey: validation.data.max_chars_per_key,
                    batchStartIndex: options.batchStartIndex,
                    batchEndIndex: options.batchEndIndex,
                }),
                warnings: [],
            })
        }), {
            name: 'get_evidence_context',
            description: 'Translation-safe evidence context using filtered_key_index only.',
            schema: evidenceContextSchema,
        }),
    ]
}

async function loadRuntime (root: string, translationStateRoot: string, manualTransFile: string, filterLanguage: string): Promise<TranslationRuntime> {
    const { manifest, manualTrans } = await ensureTranslationIndexManifest(root, manualTransFile, filterLanguage, translationStateRoot)
    const manifestByFiltered = new Map(manifest.entries.map(entry => [entry.filtered_key_index, entry]))
    const filteredByRaw = new Map(manifest.entries.map(entry => [entry.raw_key_index, entry.filtered_key_index]))

    return {
        manifest,
        manualTrans,
        manifestByFiltered,
        filteredByRaw,
    }
}

function getBatchManifestEntries (manifest: TranslationIndexManifest, batchStartIndex: number, batchEndIndex: number): FrozenTranslationIndexEntry[] {
    return manifest.entries.filter(entry => entry.filtered_key_index >= batchStartIndex && entry.filtered_key_index <= batchEndIndex)
}

function findTranslationTermMatches (
    glossary: GlossaryState,
    runtime: TranslationRuntime,
    batchStartIndex: number,
    batchEndIndex: number,
    speaker: string,
): LookupMatch[] {
    const batchEntries = getBatchManifestEntries(runtime.manifest, batchStartIndex, batchEndIndex)
    const batchTexts = batchEntries.map(entry => ({
        filteredKeyIndex: entry.filtered_key_index,
        text: runtime.manualTrans.keys[entry.raw_key_index] ?? '',
    }))
    const approvedEntriesByTerm = groupApprovedEntriesByTerm(glossary)
    const approvedEvidenceByTerm = groupApprovedEvidenceByTerm(glossary, approvedEntriesByTerm)
    const matches: LookupMatch[] = []

    for (const term of glossary.terms) {
        if (term.status !== 'active') {
            continue
        }

        const entries = approvedEntriesByTerm.get(term.term_id) ?? []
        if (entries.length === 0) {
            continue
        }

        const variants = getSourceVariants(term)
        const names = variants.map(variant => variant.text).filter(Boolean)
        const directMatches = batchTexts
            .map(item => {
                const matchedVariants = variants
                    .filter(variant => containsNormalized(item.text, variant.text))
                    .sort(compareSourceVariantMatches)
                const primaryVariant = matchedVariants[0]

                return primaryVariant
                    ? {
                        filteredKeyIndex: item.filteredKeyIndex,
                        variant: primaryVariant,
                    }
                    : null
            })
            .filter((item): item is { filteredKeyIndex: number, variant: typeof variants[number] } => item !== null)
        const directIndexes = directMatches.map(item => item.filteredKeyIndex)
        const speakerMatched = speaker.trim().length > 0 && names.some(name => containsNormalized(speaker, name) || containsNormalized(name, speaker))
        const evidence = approvedEvidenceByTerm.get(term.term_id) ?? []
        const evidenceIndexes = evidence
            .map(item => evidenceFilteredIndex(item, runtime))
            .filter((index): index is number => typeof index === 'number')
        const evidenceInBatch = evidenceIndexes.filter(index => index >= batchStartIndex && index <= batchEndIndex)

        if (directIndexes.length === 0 && !speakerMatched && evidenceInBatch.length === 0) {
            continue
        }

        const matchedIndexes = uniqueNumbers([...directIndexes, ...evidenceInBatch]).sort((left, right) => left - right)
        const distance = matchedIndexes.length > 0
            ? Math.min(...matchedIndexes.map(index => distanceFromBatch(index, batchStartIndex, batchEndIndex)))
            : Math.min(...evidenceIndexes.map(index => distanceFromBatch(index, batchStartIndex, batchEndIndex)), Number.MAX_SAFE_INTEGER)
        const translationRules = entries.filter(entry => entry.entry_type === 'translation_rule')
        const directlyMatchedVariantIds = new Set(directMatches.map(item => item.variant.variant_id))
        const applicableTranslationRules = translationRules.filter(entry => (
            (entry.applicability.source_selectors ?? []).some(selector => directlyMatchedVariantIds.has(selector.variant_id))
        ))
        const otherTranslationRules = translationRules.filter(entry => !applicableTranslationRules.includes(entry))
        const matchedSourceIndexes = formatMatchedSourceIndexes(directMatches)
        const hasDirectTranslationRule = directIndexes.length > 0 && applicableTranslationRules.length > 0
        const score = (hasDirectTranslationRule ? 1000 : 0)
            + (directIndexes.length > 0 ? 500 : 0)
            + (speakerMatched ? 250 : 0)
            + (evidenceInBatch.length > 0 ? 100 : 0)
            - Math.min(distance, 10_000)

        matches.push({
            term,
            entries,
            evidence,
            matchReason: directIndexes.length > 0 ? 'batch_text' : speakerMatched ? 'speaker' : 'evidence',
            matchedFilteredKeyIndexes: matchedIndexes,
            matchedSourceIndexes,
            score,
            distance,
            translationRules,
            applicableTranslationRules,
            otherTranslationRules,
        })
    }

    return matches.sort((left, right) => right.score - left.score || left.term.term_id.localeCompare(right.term.term_id))
}

function compareSourceVariantMatches (
    left: { index: number, text: string },
    right: { index: number, text: string },
): number {
    return normalizeText(right.text).length - normalizeText(left.text).length || left.index - right.index
}

function formatMatchedSourceIndexes (
    matches: Array<{ filteredKeyIndex: number, variant: { index: number, text: string } }>,
): MatchedSourceIndex[] {
    const byIndex = new Map<number, MatchedSourceIndex>()

    for (const match of matches) {
        const existing = byIndex.get(match.variant.index) ?? {
            index: match.variant.index,
            text: match.variant.text,
            matched_filtered_key_indexes: [],
        }
        existing.matched_filtered_key_indexes.push(match.filteredKeyIndex)
        byIndex.set(match.variant.index, existing)
    }

    return Array.from(byIndex.values()).map(item => ({
        ...item,
        matched_filtered_key_indexes: uniqueNumbers(item.matched_filtered_key_indexes).sort((left, right) => left - right),
    })).sort((left, right) => left.index - right.index)
}

function formatLookupMatch (match: LookupMatch, includeTranslationRules: boolean, enableCharacterGenderWarnings: boolean): Record<string, unknown> {
    return {
        term_id: match.term.term_id,
        source_text: match.term.source_text,
        source_text_index: 0,
        aliases: formatTermForAgent(match.term).aliases,
        term_type: match.term.term_type,
        status: match.term.status,
        merged_into: match.term.merged_into,
        match_reason: match.matchReason,
        matched_filtered_key_indexes: match.matchedFilteredKeyIndexes,
        matched_source_indexes: match.matchedSourceIndexes,
        entry_counts_by_type: countEntriesByType(match.entries),
        has_requires_context_check: match.entries.some(entry => entry.policy.requires_context_check === true),
        ...(includeTranslationRules ? {
            applicable_translation_rules: match.applicableTranslationRules.map(entry => formatTranslationEntry(entry, match.term)),
            other_translation_rules: match.otherTranslationRules.map(entry => formatTranslationEntry(entry, match.term)),
        } : {}),
        ...(match.term.term_type === 'character' ? {
            character_context: formatCharacterContext(match, enableCharacterGenderWarnings),
        } : {}),
    }
}

function formatCharacterContext (match: LookupMatch, enableCharacterGenderWarnings: boolean): Record<string, unknown> {
    const warnings = enableCharacterGenderWarnings
        ? collectCharacterContextWarnings({
            terms: [match.term],
            entries: match.entries,
            approvedOnly: true,
        })
        : []

    return {
        gender_presentations: (match.term.gender_presentations ?? [])
            .map(presentation => {
                const entry = match.entries.find(item => item.entry_id === presentation.entry_id)

                if (!entry || entry.term_id !== match.term.term_id || entry.status !== 'approved' || !isGenderPresentationEntry(entry)) {
                    return null
                }

                return {
                    value: presentation.value,
                    confidence: presentation.confidence,
                    entry_id: presentation.entry_id,
                    entry: formatTranslationEntry(entry, match.term),
                }
            })
            .filter((item): item is NonNullable<typeof item> => item !== null),
        ...(warnings.length > 0 ? { warnings } : {}),
    }
}

function collectLookupCharacterContextWarnings (matches: LookupMatch[]): TranslationToolWarning[] {
    return collectCharacterContextWarnings({
        terms: matches.map(match => match.term),
        entries: matches.flatMap(match => match.entries),
        approvedOnly: true,
    }).map(warning => ({
        code: warning.code,
        field: `term:${warning.term_id}.gender_presentations`,
        message: `${warning.message} For translation, avoid gendered Chinese address unless source context or approved entries support it.`,
    }))
}

function getTranslationTermEntries (
    glossary: GlossaryState,
    runtime: TranslationRuntime,
    input: z.infer<typeof getTranslationTermEntriesSchema>,
): TranslationTermEntriesResult {
    const cursorOffsets = decodeCursor(input.cursor)
    const nextOffsets: Record<string, number> = {}
    let hasMore = false
    const warnings: TranslationToolWarning[] = []
    const allowedTypes = input.entry_types ? new Set<EntryType>(input.entry_types) : null
    const terms = input.term_ids.map(termId => {
        const term = glossary.terms.find(item => item.term_id === termId && item.status === 'active')
        const entries = glossary.entries
            .filter(entry => entry.term_id === termId && entry.status === 'approved')
            .filter(entry => !allowedTypes || allowedTypes.has(entry.entry_type))
            .sort(compareEntriesForTranslation)
        const offset = cursorOffsets[termId] ?? 0
        const page = entries.slice(offset, offset + input.limit_per_term)
        const nextOffset = offset + page.length

        if (nextOffset < entries.length) {
            hasMore = true
            nextOffsets[termId] = nextOffset
        }

        return {
            term_id: termId,
            source_text: term?.source_text ?? null,
            entries: page.map(entry => {
                const evidence = glossary.evidence
                    .filter(item => entry.evidence_ids.includes(item.evidence_id))
                    .filter(item => evidenceFilteredIndex(item, runtime) !== null)
                const formatted = term ? formatTranslationEntry(entry, term) : formatTranslationEntryWithoutSelector(entry)
                warnings.push(...readEntrySelectorWarnings(formatted))

                if (input.include_evidence_summary === 'ids_only') {
                    return {
                        ...formatted,
                        evidence_ids: evidence.map(item => item.evidence_id),
                    }
                }

                if (input.include_evidence_summary === 'summary') {
                    return {
                        ...formatted,
                        evidence_summary: evidence
                            .map(item => formatEvidenceSummary(item, runtime))
                            .filter((item): item is NonNullable<typeof item> => item !== null),
                    }
                }

                return formatted
            }),
        }
    })

    return {
        terms,
        next_cursor: hasMore ? encodeCursor(nextOffsets) : null,
        warnings,
    }
}

function searchTranslationMemory (
    runtime: TranslationRuntime,
    translations: TranslationRecord[],
    input: z.infer<typeof searchTranslationMemorySchema>,
    options: { batchStartIndex: number, batchEndIndex: number },
): {
    total_matched: number
    returned: number
    next_cursor: string|null
    items: Array<{
        filtered_key_index: number
        source_key: string
        translation_value: string|null
        matched_fields: Array<typeof translationMemoryFieldValues[number]>
        distance_from_current_batch: number
    }>
    warnings: string[]
} {
    const fields = new Set(input.fields)
    const matcher = createTextMatcher(input.query, input.is_regex, input.case_sensitive)
    const normalizedQuery = normalizeText(input.query)
    const translationByFiltered = new Map(translations.map(item => [item.filtered_key_index, item]))
    const allMatches = runtime.manifest.entries
        .map(entry => {
            const sourceKey = runtime.manualTrans.keys[entry.raw_key_index] ?? ''
            const translation = translationByFiltered.get(entry.filtered_key_index)
            const matchedFields: Array<typeof translationMemoryFieldValues[number]> = []

            if (fields.has('source_key') && matcher(sourceKey)) {
                matchedFields.push('source_key')
            }

            if (translation && fields.has('translation_value') && matcher(translation.translation_value)) {
                matchedFields.push('translation_value')
            }

            if (matchedFields.length === 0) {
                return null
            }

            if (!translation && !input.include_unsubmitted_source_matches) {
                return null
            }

            return {
                filtered_key_index: entry.filtered_key_index,
                source_key: truncateSingleLine(sourceKey, input.max_chars_per_field),
                translation_value: translation ? truncateSingleLine(translation.translation_value, input.max_chars_per_field) : null,
                matched_fields: matchedFields,
                distance_from_current_batch: distanceFromBatch(entry.filtered_key_index, options.batchStartIndex, options.batchEndIndex),
                relevance_score: scoreTranslationMemoryMatch(input.query, normalizedQuery, sourceKey, translation?.translation_value ?? null, matchedFields),
            }
        })
        .filter((item): item is NonNullable<typeof item> => item !== null)
        .sort(compareTranslationMemoryMatches)

    const offset = decodeSingleOffsetCursor(input.cursor)
    const page = allMatches.slice(offset, offset + input.limit)
    const nextOffset = offset + page.length

    return {
        total_matched: allMatches.length,
        returned: page.length,
        next_cursor: nextOffset < allMatches.length ? encodeSingleOffsetCursor(nextOffset) : null,
        items: page.map(({ relevance_score: _relevanceScore, ...item }) => item),
        warnings: [],
    }
}

function scoreTranslationMemoryMatch (
    query: string,
    normalizedQuery: string,
    sourceKey: string,
    translationValue: string|null,
    matchedFields: Array<typeof translationMemoryFieldValues[number]>,
): number {
    let score = matchedFields.length * 100

    for (const field of matchedFields) {
        const value = field === 'source_key' ? sourceKey : translationValue
        if (!value) {
            continue
        }

        if (value === query) {
            score += 40
        }

        const normalizedValue = normalizeText(value)
        if (normalizedValue === normalizedQuery) {
            score += 30
        } else if (normalizedValue.startsWith(normalizedQuery)) {
            score += 20
        } else if (normalizedValue.includes(normalizedQuery)) {
            score += 10
        }
    }

    return score
}

function compareTranslationMemoryMatches (
    left: {
        filtered_key_index: number
        distance_from_current_batch: number
        relevance_score: number
    },
    right: {
        filtered_key_index: number
        distance_from_current_batch: number
        relevance_score: number
    },
): number {
    return right.relevance_score - left.relevance_score
        || left.distance_from_current_batch - right.distance_from_current_batch
        || left.filtered_key_index - right.filtered_key_index
}

async function submitTranslationBatch (
    translationStateRoot: string,
    runtime: TranslationRuntime,
    batchEntries: FrozenTranslationIndexEntry[],
    translations: { filtered_key_index: number, translation_value: string }[],
    options: { batchId?: string, submittedBy?: string },
): Promise<unknown> {
    const batchIndexes = batchEntries.map(entry => entry.filtered_key_index)
    const batchIndexSet = new Set(batchIndexes)
    const seen = new Set<number>()
    const errors: { field: string, message: string }[] = []
    const existingTranslations = await readTranslationState(translationStateRoot, state => state.translations)
    const existingIndexes = new Set(existingTranslations.map(item => item.filtered_key_index))
    const requiredIndexes = batchIndexes.filter(filteredIndex => !existingIndexes.has(filteredIndex))

    for (const [index, item] of translations.entries()) {
        if (!batchIndexSet.has(item.filtered_key_index)) {
            errors.push({ field: `translations[${index}].filtered_key_index`, message: 'filtered_key_index is outside the current batch.' })
        }

        if (seen.has(item.filtered_key_index)) {
            errors.push({ field: `translations[${index}].filtered_key_index`, message: 'Duplicate filtered_key_index in submit_translation_batch.' })
        }

        seen.add(item.filtered_key_index)
    }

    for (const filteredIndex of requiredIndexes) {
        if (!seen.has(filteredIndex)) {
            errors.push({ field: 'translations', message: `Missing translation for filtered_key_index ${filteredIndex}.` })
        }
    }

    for (const filteredIndex of seen) {
        if (existingIndexes.has(filteredIndex)) {
            errors.push({ field: 'translations', message: `filtered_key_index ${filteredIndex} has already been submitted.` })
        }
    }

    for (const [index, item] of translations.entries()) {
        const manifestEntry = runtime.manifestByFiltered.get(item.filtered_key_index)
        const sourceKey = manifestEntry ? runtime.manualTrans.keys[manifestEntry.raw_key_index] ?? '' : ''
        const protectionErrors = validateProtectedTokens(sourceKey, item.translation_value)

        errors.push(...protectionErrors.map(message => ({
            field: `translations[${index}].translation_value`,
            message,
        })))
    }

    if (errors.length > 0) {
        return validationErrorOutput(errors)
    }

    const now = new Date().toISOString()
    const records: TranslationRecord[] = translations.map(item => {
        const manifestEntry = runtime.manifestByFiltered.get(item.filtered_key_index)!
        return {
            filtered_key_index: item.filtered_key_index,
            source_key_hash: manifestEntry.source_key_hash,
            translation_value: item.translation_value,
            submitted_at: now,
            ...(options.submittedBy ? { submitted_by: options.submittedBy } : {}),
            ...(options.batchId ? { batch_id: options.batchId } : {}),
        }
    })

    await updateTranslationState(translationStateRoot, state => {
        state.translations.push(...records)
        state.translations.sort((left, right) => left.filtered_key_index - right.filtered_key_index)
        state.meta.updated_at = now
    })

    return {
        ok: true,
        submitted_count: records.length,
        filtered_key_indexes: records.map(item => item.filtered_key_index),
        retry_required: false,
        warnings: [],
    }
}

async function queryTranslationGlossaryTerms (
    glossary: GlossaryState,
    runtime: TranslationRuntime,
    input: z.infer<typeof queryGlossaryTermsSchema>,
): Promise<{ matched_terms: TranslationGlossaryTermMatch[], warnings: string[] }> {
    const matchedTerms: TranslationGlossaryTermMatch[] = []
    const seenTermIds = new Set<string>()
    const yieldController = createYieldController()

    for (const query of input.queries) {
        for (const mode of query.match_modes) {
            const matcher = createTermMatcher(query.text, mode, query.case_sensitive)

            for (const [termIndex, term] of glossary.terms.entries()) {
                await yieldController.maybeYield(termIndex)
                if (seenTermIds.has(term.term_id) || term.source_language !== input.source_language || term.status !== 'active') {
                    continue
                }

                if (input.term_statuses && !input.term_statuses.includes('active')) {
                    continue
                }

                const matchedText = matcher(term)

                if (!matchedText) {
                    continue
                }

                const entries = input.include_entries
                    ? approvedEntriesForTerm(glossary, term.term_id, input.entry_statuses ?? null)
                    : []
                const evidence = input.include_evidence
                    ? approvedEvidenceForEntries(glossary, approvedEntriesForTerm(glossary, term.term_id, input.entry_statuses ?? null))
                    : []

                matchedTerms.push({
                    term: formatTranslationTerm(term),
                    match: {
                        query_text: query.text,
                        match_mode: mode,
                        matched_text: matchedText,
                    },
                    entries: entries.map(entry => formatTranslationEntry(entry, term)),
                    evidence: evidence
                        .map(item => formatEvidenceSummary(item, runtime))
                        .filter((item): item is EvidenceSummary => item !== null),
                })
                seenTermIds.add(term.term_id)

                if (matchedTerms.length >= input.limit) {
                    return { matched_terms: matchedTerms, warnings: [] }
                }
            }
        }
    }

    return { matched_terms: matchedTerms, warnings: [] }
}

async function searchTranslationGlossaryEntries (
    glossary: GlossaryState,
    runtime: TranslationRuntime,
    input: z.infer<typeof searchGlossaryEntriesSchema>,
): Promise<{ matched_entries: TranslationGlossaryEntryMatch[], warnings: string[] }> {
    const matchedEntries: TranslationGlossaryEntryMatch[] = []
    const yieldController = createYieldController()

    for (const [entryIndex, entry] of glossary.entries.entries()) {
        await yieldController.maybeYield(entryIndex)
        if (entry.status !== 'approved') {
            continue
        }

        const term = glossary.terms.find(item => item.term_id === entry.term_id)

        if (!term || term.status !== 'active') {
            continue
        }

        if (!entryMatchesFilter(entry, input.entry_filter ?? null) || !termMatchesFilter(term, input.term_filter ?? null)) {
            continue
        }

        const evidence = input.include_evidence ? approvedEvidenceForEntries(glossary, [entry]) : []
        matchedEntries.push({
            entry: formatTranslationEntry(entry, term),
            ...(input.include_term ? { term: formatTranslationTerm(term) } : {}),
            evidence: evidence
                .map(item => formatEvidenceSummary(item, runtime))
                .filter((item): item is EvidenceSummary => item !== null),
        })

        if (matchedEntries.length >= input.limit) {
            break
        }
    }

    return { matched_entries: matchedEntries, warnings: [] }
}

function createEvidenceContexts (
    evidenceItems: GlossaryEvidence[],
    runtime: TranslationRuntime,
    evidenceIds: string[],
    options: {
        before: number
        after: number
        maxCharsPerKey: number
        batchStartIndex: number
        batchEndIndex: number
    },
): Record<string, unknown>[] {
    return evidenceIds.map(evidenceId => {
        const evidence = evidenceItems.find(item => item.evidence_id === evidenceId)
        const filteredIndex = evidence ? evidenceFilteredIndex(evidence, runtime) : null

        if (!evidence || filteredIndex === null) {
            return {
                evidence_id: evidenceId,
                error: 'Evidence cannot be mapped to frozen filtered_key_index.',
            }
        }

        const startIndex = Math.max(0, filteredIndex - options.before)
        const endIndex = filteredIndex + options.after
        const contextItems = runtime.manifest.entries
            .filter(entry => entry.filtered_key_index >= startIndex && entry.filtered_key_index <= endIndex)
            .map(entry => ({
                filtered_key_index: entry.filtered_key_index,
                source_key: truncate(runtime.manualTrans.keys[entry.raw_key_index] ?? '', options.maxCharsPerKey),
            }))

        return {
            evidence_id: evidence.evidence_id,
            term_id: evidence.term_id,
            entry_id: evidence.entry_id,
            filtered_key_index: filteredIndex,
            quote: evidence.quote,
            reason: evidence.reason,
            distance_from_current_batch: distanceFromBatch(filteredIndex, options.batchStartIndex, options.batchEndIndex),
            context: contextItems,
        }
    })
}

function groupApprovedEntriesByTerm (glossary: GlossaryState): Map<string, GlossaryEntry[]> {
    const result = new Map<string, GlossaryEntry[]>()

    for (const entry of glossary.entries) {
        if (entry.status !== 'approved') {
            continue
        }

        const list = result.get(entry.term_id) ?? []
        list.push(entry)
        result.set(entry.term_id, list)
    }

    return result
}

function groupApprovedEvidenceByTerm (
    glossary: GlossaryState,
    entriesByTerm: Map<string, GlossaryEntry[]>,
): Map<string, GlossaryEvidence[]> {
    const approvedEntryIds = new Set(Array.from(entriesByTerm.values()).flat().map(entry => entry.entry_id))
    const result = new Map<string, GlossaryEvidence[]>()

    for (const evidence of glossary.evidence) {
        if (!approvedEntryIds.has(evidence.entry_id)) {
            continue
        }

        const list = result.get(evidence.term_id) ?? []
        list.push(evidence)
        result.set(evidence.term_id, list)
    }

    return result
}

function approvedEntriesForTerm (glossary: GlossaryState, termId: string, requestedStatuses: EntryStatus[]|null): GlossaryEntry[] {
    if (requestedStatuses && !requestedStatuses.includes('approved')) {
        return []
    }

    return glossary.entries.filter(entry => entry.term_id === termId && entry.status === 'approved')
}

function approvedEvidenceForEntries (glossary: GlossaryState, entries: GlossaryEntry[]): GlossaryEvidence[] {
    const entryIds = new Set(entries.map(entry => entry.entry_id))
    return glossary.evidence.filter(evidence => entryIds.has(evidence.entry_id))
}

function countEntriesByType (entries: GlossaryEntry[]): Record<EntryType, number> {
    return {
        translation_rule: entries.filter(entry => entry.entry_type === 'translation_rule').length,
        continuity: entries.filter(entry => entry.entry_type === 'continuity').length,
        style: entries.filter(entry => entry.entry_type === 'style').length,
        fact: entries.filter(entry => entry.entry_type === 'fact').length,
    }
}

function compareEntriesForTranslation (left: GlossaryEntry, right: GlossaryEntry): number {
    return entryTypeRank(left.entry_type) - entryTypeRank(right.entry_type) || left.entry_id.localeCompare(right.entry_id)
}

function entryTypeRank (entryType: EntryType): number {
    if (entryType === 'translation_rule') {
        return 0
    }
    if (entryType === 'continuity') {
        return 1
    }
    if (entryType === 'style') {
        return 2
    }
    return 3
}

function isGenderPresentationEntry (entry: Pick<GlossaryEntry, 'entry_type'>): boolean {
    return entry.entry_type === 'fact' || entry.entry_type === 'style' || entry.entry_type === 'continuity'
}

function formatTranslationTerm (term: GlossaryTerm): Record<string, unknown> & { term_id: string } {
    return formatTermForAgent(term)
}

function formatTranslationEntry (entry: GlossaryEntry, term: GlossaryTerm): Record<string, unknown> {
    const selectorFields = selectorsToAgentApplicabilityFields(term, entry.applicability.source_selectors)
    const {
        source_selectors: _sourceSelectors,
        ...applicability
    } = entry.applicability

    return {
        entry_id: entry.entry_id,
        term_id: entry.term_id,
        entry_type: entry.entry_type,
        content: entry.content,
        ...(entry.target ? { target: entry.target } : {}),
        applicability: {
            ...applicability,
            ...(selectorFields.source_variant_indexes ? { source_variant_indexes: selectorFields.source_variant_indexes } : {}),
            ...(selectorFields.source_variant_texts ? { source_variant_texts: selectorFields.source_variant_texts } : {}),
        },
        policy: {
            strength: entry.policy.strength,
            requires_context_check: entry.policy.requires_context_check ?? false,
            ...(entry.policy.notes ? { notes: entry.policy.notes } : {}),
        },
        status: entry.status,
        ...(selectorFields.warnings.length > 0 ? {
            source_selector_warnings: selectorFields.warnings.map(warning => ({
                ...warning,
                entry_id: entry.entry_id,
            })),
        } : {}),
    }
}

function formatTranslationEntryWithoutSelector (entry: GlossaryEntry): Record<string, unknown> {
    const {
        source_selectors: _sourceSelectors,
        ...applicability
    } = entry.applicability

    return {
        entry_id: entry.entry_id,
        term_id: entry.term_id,
        entry_type: entry.entry_type,
        content: entry.content,
        ...(entry.target ? { target: entry.target } : {}),
        applicability,
        policy: {
            strength: entry.policy.strength,
            requires_context_check: entry.policy.requires_context_check ?? false,
            ...(entry.policy.notes ? { notes: entry.policy.notes } : {}),
        },
        status: entry.status,
    }
}

function formatEvidenceSummary (evidence: GlossaryEvidence, runtime: TranslationRuntime): EvidenceSummary|null {
    const filteredIndex = evidenceFilteredIndex(evidence, runtime)

    if (filteredIndex === null) {
        return null
    }

    return {
        evidence_id: evidence.evidence_id,
        term_id: evidence.term_id,
        entry_id: evidence.entry_id,
        filtered_key_index: filteredIndex,
        quote: evidence.quote,
    }
}

function evidenceFilteredIndex (evidence: GlossaryEvidence, runtime: TranslationRuntime): number|null {
    const sourceFileId = evidence.source_ref?.source_file_id ?? evidence.source_ref?.file_id
    const rawIndex = evidence.source_ref?.key_index

    if (typeof rawIndex !== 'number') {
        return null
    }

    const keyHash = evidence.source_ref?.key_hash
    const sourceKey = runtime.manualTrans.keys[rawIndex]

    if (keyHash) {
        return typeof sourceKey === 'string' && normalizeKeyHash(keyHash) === sha256Text(sourceKey)
            ? runtime.filteredByRaw.get(rawIndex) ?? null
            : null
    }

    if (sourceFileId && normalizeProjectFileId(sourceFileId) !== normalizeProjectFileId(runtime.manifest.source_file_id)) {
        return null
    }

    return runtime.filteredByRaw.get(rawIndex) ?? null
}

function readEntrySelectorWarnings (entry: Record<string, unknown>): TranslationToolWarning[] {
    const sourceSelectorWarnings = entry.source_selector_warnings

    return Array.isArray(sourceSelectorWarnings)
        ? sourceSelectorWarnings.filter((warning): warning is TranslationToolWarning => (
            typeof warning === 'object'
            && warning !== null
            && typeof (warning as { code?: unknown }).code === 'string'
            && typeof (warning as { field?: unknown }).field === 'string'
            && typeof (warning as { message?: unknown }).message === 'string'
        ))
        : []
}

function createTermMatcher (text: string, mode: typeof matchModeValues[number], caseSensitive: boolean): (term: GlossaryTerm) => string|null {
    if (mode === 'regex') {
        const regex = new RegExp(text, caseSensitive ? '' : 'i')
        return term => {
            for (const value of [term.source_text, ...getAliasTexts(term)]) {
                regex.lastIndex = 0
                if (regex.test(value)) {
                    return value
                }
            }

            return null
        }
    }

    const query = caseSensitive ? text : normalizeText(text)

    return term => {
        for (const value of [term.source_text, ...getAliasTexts(term)]) {
            const candidate = caseSensitive ? value : normalizeText(value)

            if (
                mode === 'exact' && candidate === query
                || mode === 'alias' && getAliasTexts(term).includes(value) && candidate === query
                || mode === 'compound' && (candidate.includes(query) || query.includes(candidate))
                || mode === 'fuzzy' && similarity(candidate, query) >= 0.72
            ) {
                return value
            }
        }

        return null
    }
}

function entryMatchesFilter (entry: GlossaryEntry, filter: z.infer<typeof searchGlossaryEntriesSchema>['entry_filter']): boolean {
    if (!filter) {
        return true
    }

    if (filter.entry_types && !filter.entry_types.includes(entry.entry_type)) {
        return false
    }

    if (filter.entry_statuses && !filter.entry_statuses.includes('approved')) {
        return false
    }

    if (filter.applies_to && !filter.applies_to.includes(entry.applicability.applies_to)) {
        return false
    }

    if (filter.target_language && entry.target?.target_language !== filter.target_language) {
        return false
    }

    if (filter.policy_strength && !filter.policy_strength.includes(entry.policy.strength)) {
        return false
    }

    if (filter.domain && entry.applicability.domain !== filter.domain) {
        return false
    }

    if (filter.text) {
        const matcher = createTextMatcher(filter.text, filter.is_regex ?? false, filter.case_sensitive ?? false)
        const fields = filter.text_fields ?? ['content.summary', 'content.description']
        const values = fields.flatMap(field => getEntryTextFieldValues(entry, field))

        if (!values.some(value => matcher(value))) {
            return false
        }
    }

    return true
}

function termMatchesFilter (term: GlossaryTerm, filter: z.infer<typeof searchGlossaryEntriesSchema>['term_filter']): boolean {
    if (!filter) {
        return true
    }

    if (filter.source_language && term.source_language !== filter.source_language) {
        return false
    }

    if (filter.term_types && !filter.term_types.includes(term.term_type)) {
        return false
    }

    return !(filter.term_statuses && !filter.term_statuses.includes('active'))
}

function getEntryTextFieldValues (entry: GlossaryEntry, field: string): string[] {
    if (field === 'content.summary') {
        return [entry.content.summary]
    }
    if (field === 'content.description') {
        return [entry.content.description]
    }
    if (field === 'target.preferred_translation') {
        return entry.target?.preferred_translation ? [entry.target.preferred_translation] : []
    }
    if (field === 'target.alternative_translations') {
        return entry.target?.alternative_translations ?? []
    }
    if (field === 'target.forbidden_translations') {
        return entry.target?.forbidden_translations ?? []
    }
    if (field === 'policy.notes') {
        return entry.policy.notes ?? []
    }
    if (field === 'applicability.applies_when') {
        return entry.applicability.applies_when ?? []
    }
    if (field === 'applicability.does_not_apply_when') {
        return entry.applicability.does_not_apply_when ?? []
    }
    return []
}

// Differs from glossary text search: non-regex matching normalizes before matching.
function createTextMatcher (text: string, isRegex: boolean, caseSensitive: boolean): (value: string) => boolean {
    if (isRegex) {
        const regex = new RegExp(text, caseSensitive ? '' : 'i')
        return value => {
            regex.lastIndex = 0
            return regex.test(value)
        }
    }

    const needle = caseSensitive ? text : normalizeText(text)
    return value => (caseSensitive ? value : normalizeText(value)).includes(needle)
}

function analyzeProtectionRisks (sourceKey: string): Record<string, unknown> {
    const protectedTokens = extractProtectedTokens(sourceKey)
    const newlineCount = (sourceKey.match(/\n/g) ?? []).length

    return {
        protected_tokens: protectedTokens,
        has_newlines: newlineCount > 0,
        newline_count: newlineCount,
        has_placeholder_like_text: protectedTokens.length > 0,
    }
}

function validateProtectedTokens (sourceKey: string, translationValue: string): string[] {
    const errors: string[] = []
    const sourceTokens = extractProtectedTokens(sourceKey)

    for (const token of sourceTokens) {
        if (!translationValue.includes(token)) {
            errors.push(`Translation must preserve protected token ${token}.`)
        }
    }

    const sourceNewlines = (sourceKey.match(/\n/g) ?? []).length
    const translationNewlines = (translationValue.match(/\n/g) ?? []).length

    if (sourceNewlines !== translationNewlines) {
        errors.push('Translation must preserve newline count.')
    }

    return errors
}

function extractProtectedTokens (sourceKey: string): string[] {
    const patterns = [
        /\\[A-Za-z]{1,4}\[[^\]]+]/g,
        /\\[A-Za-z]{1,4}/g,
        /%[0-9]+/g,
        /%[sdif]/g,
        /<[^>]+>/g,
        /\[[A-Za-z][^\]\n]{0,80}]/g,
        /\{[A-Za-z0-9_$.-]+}/g,
    ]
    const tokens: string[] = []

    for (const pattern of patterns) {
        for (const match of sourceKey.matchAll(pattern)) {
            tokens.push(match[0])
        }
    }

    return uniqueStrings(tokens)
}

function validatePreflightText (input: z.infer<typeof submitTranslationPreflightSchema>): { field: string, message: string }[] {
    // Currently dead code: submit_translation_preflight does not call this regex screen, but it may be re-enabled later.
    const errors: { field: string, message: string }[] = []
    const disallowed = /\b(batch_size|batch size|parallel_agents|parallel agents|Term|Entry|Evidence|preferred_translation|forbidden_translations)\b|译为|翻译为|禁译/iu

    for (const [field, value] of Object.entries(input)) {
        if (!Array.isArray(value)) {
            continue
        }

        value.forEach((item, index) => {
            if (disallowed.test(item)) {
                errors.push({ field: `${field}[${index}]`, message: 'submit_translation_preflight must not include concrete translations, glossary records, or scheduling strategy.' })
            }
        })
    }

    return errors
}

function validateQueryRegexes (queries: z.infer<typeof queryGlossaryTermsSchema>['queries']): { field: string, message: string }[] {
    const errors: { field: string, message: string }[] = []

    queries.forEach((query, index) => {
        if (query.match_modes.includes('regex')) {
            errors.push(...validateRegexText(`queries[${index}].text`, query.text))
        }
    })

    return errors
}

function validateSearchRegex (filter: z.infer<typeof searchGlossaryEntriesSchema>['entry_filter']): { field: string, message: string }[] {
    if (!filter?.is_regex || !filter.text) {
        return []
    }

    return validateRegexText('entry_filter.text', filter.text)
}

// Same regex compile check as glossary tools; returns English validation errors.
function validateRegexText (field: string, text: string): { field: string, message: string }[] {
    try {
        new RegExp(text)
        return []
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return [{ field, message: `Invalid regular expression: ${message}` }]
    }
}

function encodeCursor (offsets: Record<string, number>): string {
    return Buffer.from(JSON.stringify(offsets), 'utf8').toString('base64url')
}

function encodeSingleOffsetCursor (offset: number): string {
    return encodeCursor({ offset })
}

function decodeSingleOffsetCursor (cursor: string|undefined): number {
    return decodeCursor(cursor).offset ?? 0
}

function decodeCursor (cursor: string|undefined): Record<string, number> {
    if (!cursor) {
        return {}
    }

    try {
        const parsed: unknown = JSON.parse(Buffer.from(cursor, 'base64url').toString('utf8'))
        if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
            return {}
        }

        return Object.fromEntries(Object.entries(parsed).filter((entry): entry is [string, number] => (
            typeof entry[0] === 'string' && typeof entry[1] === 'number' && Number.isInteger(entry[1]) && entry[1] >= 0
        )))
    } catch {
        return {}
    }
}

function containsNormalized (haystack: string, needle: string): boolean {
    return normalizeText(haystack).includes(normalizeText(needle))
}

// Same text normalization contract as glossary source-term matching.
function normalizeText (value: string): string {
    return value.normalize('NFKC').trim().replace(/\s+/g, ' ').toLocaleLowerCase()
}

// Same tool payload string handling as glossary tools: trim, drop empty values, keep first-seen order.
function uniqueStrings (values: string[]): string[] {
    return Array.from(new Set(values.map(value => value.trim()).filter(Boolean)))
}

function uniqueNumbers (values: number[]): number[] {
    return Array.from(new Set(values))
}

function isTranslationBatchSubmissionResult (value: unknown): value is {
    ok: true
    submitted_count: number
    filtered_key_indexes: number[]
} {
    return typeof value === 'object'
        && value !== null
        && (value as { ok?: unknown }).ok === true
        && typeof (value as { submitted_count?: unknown }).submitted_count === 'number'
        && Array.isArray((value as { filtered_key_indexes?: unknown }).filtered_key_indexes)
        && (value as { filtered_key_indexes: unknown[] }).filtered_key_indexes.every(item => typeof item === 'number')
}

function distanceFromBatch (filteredIndex: number, batchStartIndex: number, batchEndIndex: number): number {
    if (filteredIndex >= batchStartIndex && filteredIndex <= batchEndIndex) {
        return 0
    }

    if (filteredIndex < batchStartIndex) {
        return batchStartIndex - filteredIndex
    }

    return filteredIndex - batchEndIndex
}

// Same similarity helper as glossaryTools.similarity.
function similarity (left: string, right: string): number {
    if (left === right) {
        return 1
    }

    const longest = Math.max(left.length, right.length)

    if (longest === 0) {
        return 1
    }

    return 1 - levenshteinDistance(left, right) / longest
}

function zodErrorOutput (error: z.ZodError): unknown {
    return validationErrorOutput(error.errors.map(item => ({
        field: item.path.join('.') || '(root)',
        message: item.message,
    })))
}

// Differs from glossary tool validation: retryable by default.
function validationErrorOutput (errors: { field: string, message: string }[]): unknown {
    return {
        ok: false,
        code: 'validation_error',
        errors,
        retry_required: true,
    }
}

function normalizeKeyHash (value: string): string {
    return value.startsWith('sha256:') ? value.slice('sha256:'.length) : value
}

// Same raw SHA-256 helper as glossaryTools.sha256Text.
function sha256Text (value: string): string {
    return createHash('sha256').update(value).digest('hex')
}

// Differs from taskStore.truncateSingleLine: preserves spaces, then escapes newlines.
function truncateSingleLine (value: string, maxLength: number): string {
    return truncate(value, maxLength).replace(/\n/g, '\\n')
}
