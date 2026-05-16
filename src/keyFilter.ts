import { readFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

import { DomUtils, parseDocument } from 'htmlparser2'

import { createYieldController } from './eventLoop.js'
import { resolveLanguageModelPath, resolveMediaPipeWasmPath } from './runtimeResources.js'

let languageDetectorPromise: Promise<LanguageDetectorInstance>|null = null
let mediaPipeTasksPromise: Promise<MediaPipeTasksText>|null = null

export const DEFAULT_KEY_FILTER_LANGUAGE = 'auto'

export type KeyFilterOptions = {
    filter?: boolean
    filterLanguage?: string
}

export type KeyFilter = {
    enabled: boolean
    language: string
}

export type FilteredManualTransKey = {
    key: string
    originalIndex: number
    filteredIndex: number
}

type KeyFilterItem = {
    text_index: number
    source_text: string
    translation_status: number
    lang_code: [string, number, string[]]|null
}

type LanguageStats = Array<[string, number, number]>

type MediaPipeLanguage = {
    languageCode: string
    probability: number
}

type LanguageDetectorInstance = {
    detect: (sourceText: string) => { languages: MediaPipeLanguage[] }
}

type MediaPipeTasksText = {
    FilesetResolver: {
        forTextTasks: (path: string) => Promise<unknown>
    }
    LanguageDetector: {
        createFromOptions: (fileset: unknown, options: unknown) => Promise<LanguageDetectorInstance>
    }
}

const translationStatus = Object.freeze({
    UNTRANSLATED: 0,
    EXCLUDED: 7,
})

const PUNCTUATION_CHARS = new Set(Array.from(
    '  !"#$%&\'()*+,-./，。:;<=>?@[\\]^_`{|}~—・？↑←↓→「」『』【】《》！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠',
))

const EXCLUDE_PREFIX = Object.freeze([
    'MapData/',
    'SE/',
    'BGS',
    '0=',
    'BGM/',
    'FIcon/',
    '<input type=',
    'width:',
    '<div ',
    'EV0',
    '\\img',
])

const EXCLUDE_FILE_SUFFIX = new Set([
    '.mp3',
    '.wav',
    '.png',
    '.jpg',
    '.gif',
    '.rar',
    '.zip',
    '.json',
    '.ogg',
    '.txt',
    '.mps',
    '.woff',
    '.webp',
    '.jpg)',
    '.doc',
    '.html',
    '.bmp',
    '.pic',
    '.aac',
    '.flac',
    '.avi',
    '.py',
    '.c',
    '.cpp',
    '.js',
    '.java',
    '.css',
    '.xml',
    '.jpeg',
    '.mov',
    '.mkv',
    '.flv',
])

const VARIOUS_LETTERS_RANGE = 'a-zA-Z\\uFF21-\\uFF3A\\uFF41-\\uFF5A'
const HAS_UNUSUAL_ENG_REGEX = new RegExp(
    '^(?:' +
    `(?=.*[_$])(?=.*[${VARIOUS_LETTERS_RANGE}\\d])[${VARIOUS_LETTERS_RANGE}\\d_$]+|` +
    `\\[[${VARIOUS_LETTERS_RANGE}\\d_$]+]|` +
    `\\{[${VARIOUS_LETTERS_RANGE}\\d_$]+}|` +
    `(?=.*[${VARIOUS_LETTERS_RANGE}])(?=.*\\d)[${VARIOUS_LETTERS_RANGE}\\d]*|` +
    'dummy' +
    ')$',
)

const CLEAN_TEXT_PATTERN = new RegExp(
    `\\\\{1,2}[${VARIOUS_LETTERS_RANGE}]{1,2}\\[\\d+]|` +
    'if\\(.{0,16}[vs]\\[\\d+].{0,16}\\)|' +
    '\\\\n|' +
    `[${VARIOUS_LETTERS_RANGE}]+\\d+(?!\\d*[${VARIOUS_LETTERS_RANGE}])`,
    'g',
)

const JS_BUILTIN_OBJECTS = [
    'Math',
    'Array',
    'Object',
    'String',
    'Number',
    'Boolean',
    'Function',
    'JSON',
    'Date',
    'Promise',
    'RegExp',
    'Symbol',
    'BigInt',
    'Error',
    'TypeError',
    'SyntaxError',
    'ReferenceError',
    'RangeError',
    'URIError',
    'EvalError',
    'Map',
    'Set',
    'WeakMap',
    'WeakSet',
    'Proxy',
    'Reflect',
    'ArrayBuffer',
    'DataView',
    'window',
    'document',
    'navigator',
    'location',
    'history',
    'screen',
    'localStorage',
    'sessionStorage',
    'console',
    'Node',
    'Element',
    'HTMLElement',
    'Event',
    'EventTarget',
    'NodeList',
    'HTMLCollection',
    'XMLHttpRequest',
    'WebSocket',
    'Worker',
    'Request',
    'Response',
    'Headers',
    'URL',
    'URLSearchParams',
    'FormData',
    'Blob',
    'File',
    'Intl',
    'performance',
    'crypto',
    'Vue',
    'React',
    'Component',
    'process',
    'Buffer',
    'module',
    'exports',
    'require',
    'global',
    '__dirname',
    '__filename',
]

const JS_BUILTINS_PATTERN = JS_BUILTIN_OBJECTS.map(escapeRegex).join('|')
const JS_VAR_PATTERN = new RegExp(
    `(?<![a-zA-Z0-9_$])(?:[a-z_$][a-z0-9_$]*|${JS_BUILTINS_PATTERN})(?:\\.[a-z_$][a-z0-9_$]*)+`,
    'g',
)
const TAG_STYLE_PATTERN = /\[([a-zA-Z]\w*)\s*(\s+\w+\s*=\s*(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^\s\]]+))*\s*]/g
const EXTRACT_TAG_ATTR_VALUE_PATTERN = /\w+\s*=(["'])((?:\\.|(?!\1).)*)\1/g
const XML_COLON_TAG_PATTERN = /<[^:]+:(.*?)>/gu
const MOJIBAKE_SIGNAL_PATTERN = /(?:縺|譁|荳|蜊|邱|繧|螟|逕|蜿|髱|驥|謌|譛|窶){2,}/u
const NON_LATIN_ISO_CODES = new Set([
    'zh',
    'ja',
    'ko',
    'ar',
    'ru',
    'he',
    'hi',
    'th',
    'bn',
    'el',
    'hy',
    'ka',
    'ta',
    'ml',
    'ur',
    'fa',
])

const LANGUAGE_NAME_TO_CODE: Record<string, string> = Object.freeze({
    chinese_simplified: 'zh',
    chinese_traditional: 'zh-Hant',
    english: 'en',
    spanish: 'es',
    french: 'fr',
    german: 'de',
    indonesian: 'id',
    vietnamese: 'vi',
    korean: 'ko',
    russian: 'ru',
    thai: 'th',
    japanese: 'ja',
})

const LANGUAGE_CODE_TO_NAME: Record<string, string> = Object.freeze({
    zh: 'chinese_simplified',
    'zh-cn': 'chinese_simplified',
    'zh-Hans': 'chinese_simplified',
    'zh-tw': 'chinese_traditional',
    yue: 'chinese_traditional',
    'zh-Hant': 'chinese_traditional',
    en: 'english',
    es: 'spanish',
    fr: 'french',
    de: 'german',
    id: 'indonesian',
    in: 'indonesian',
    vi: 'vietnamese',
    ko: 'korean',
    ru: 'russian',
    th: 'thai',
    ja: 'japanese',
})

export function normalizeKeyFilterOptions (options: KeyFilterOptions): KeyFilter {
    return {
        enabled: options.filter ?? true,
        language: normalizeFilterLanguage(options.filterLanguage),
    }
}

export async function filterManualTransKeyItems (root: string, keys: string[], filterLanguage: string): Promise<FilteredManualTransKey[]> {
    const items = keys.map((key, index) => createKeyFilterItem(key, index + 1))
    await filterGeneralKeyText(items)

    try {
        await detectKeyItemLanguages(root, items)
        const languageStats = await buildKeyLanguageStats(root, items)
        const lcLanguageStats = buildLowConfidenceKeyLanguageStats(items, languageStats)
        filterKeyItemsByLanguage(items, {
            sourceLanguage: filterLanguage,
            targetLanguage: 'chinese_simplified',
            languageStats,
            lcLanguageStats,
        })
    } catch {
        // Keep the deterministic general filter even when the optional language model is unavailable.
    }

    return items
        .filter(item => item.translation_status !== translationStatus.EXCLUDED)
        .map((item, filteredIndex) => ({
            key: item.source_text,
            originalIndex: item.text_index - 1,
            filteredIndex,
        }))
}

function normalizeFilterLanguage (language: string|undefined): string {
    const normalized = language?.trim()
    return normalized && normalized.length > 0 ? normalized : DEFAULT_KEY_FILTER_LANGUAGE
}

function createKeyFilterItem (sourceText: string, textIndex: number): KeyFilterItem {
    return {
        text_index: textIndex,
        source_text: sourceText,
        translation_status: translationStatus.UNTRANSLATED,
        lang_code: null,
    }
}

async function filterGeneralKeyText (items: KeyFilterItem[]): Promise<void> {
    const yieldController = createYieldController()

    for (const [index, item] of items.entries()) {
        await yieldController.maybeYield(index)
        applyGeneralKeyFilterToItem(item)
    }
}

function applyGeneralKeyFilterToItem (item: KeyFilterItem): void {
    const sourceText = item.source_text

    if (isDigitString(sourceText) || sourceText.trim() === '') {
        item.translation_status = translationStatus.EXCLUDED
        return
    }

    if (['\n', '\\n', '\r', '\\r'].includes(sourceText.trim())) {
        item.translation_status = translationStatus.EXCLUDED
        return
    }

    if (isPunctuationString(sourceText)) {
        item.translation_status = translationStatus.EXCLUDED
        return
    }

    if (EXCLUDE_FILE_SUFFIX.has(getFileSuffix(sourceText.trimEnd()))) {
        item.translation_status = translationStatus.EXCLUDED
        return
    }

    if (isGarbledOrBinaryLikeText(sourceText)) {
        item.translation_status = translationStatus.EXCLUDED
        return
    }

    if (EXCLUDE_PREFIX.some(prefix => sourceText.startsWith(prefix))) {
        item.translation_status = translationStatus.EXCLUDED
    }
}

async function detectKeyItemLanguages (root: string, items: KeyFilterItem[]): Promise<void> {
    const detector = await getLanguageDetector(root)
    const yieldController = createYieldController()

    for (const [index, item] of items.entries()) {
        await yieldController.maybeYield(index)
        if (item.translation_status === translationStatus.EXCLUDED) {
            continue
        }

        const [langs, score] = detectLanguageWithMediaPipe(detector, item.source_text)

        if (score > 0.0) {
            item.lang_code = [langs[0], score, langs.slice(1)]
        } else {
            item.translation_status = translationStatus.EXCLUDED
        }
    }
}

async function getLanguageDetector (root: string): Promise<LanguageDetectorInstance> {
    if (!languageDetectorPromise) {
        languageDetectorPromise = (async () => {
            const { FilesetResolver, LanguageDetector } = await importMediaPipeTasksText()
            const modelPath = resolveLanguageModelPath(root)
            const wasmBasePath = await resolveDefaultWasmPath()
            const wasmFileset = await FilesetResolver.forTextTasks(pathToFileURL(`${wasmBasePath}${path.sep}`).href)
            const modelAssetBuffer = new Uint8Array(await readFile(modelPath))

            return LanguageDetector.createFromOptions(wasmFileset, {
                baseOptions: {
                    modelAssetBuffer,
                },
                maxResults: 4,
                scoreThreshold: 0.0001,
                categoryDenylist: ['hmn'],
            })
        })()
    }

    return languageDetectorPromise
}

async function importMediaPipeTasksText (): Promise<MediaPipeTasksText> {
    setupMediaPipeNodeRuntime()
    mediaPipeTasksPromise ??= import('@mediapipe/tasks-text') as Promise<MediaPipeTasksText>
    return mediaPipeTasksPromise
}

type MediaPipeGlobal = {
    self?: Record<string, unknown>
    importScripts?: () => never
    OffscreenCanvas?: unknown
}

function setupMediaPipeNodeRuntime (): void {
    const mediaPipeGlobal = globalThis as unknown as MediaPipeGlobal
    mediaPipeGlobal.self ??= mediaPipeGlobal as unknown as Record<string, unknown>
    mediaPipeGlobal.importScripts ??= () => {
        throw new TypeError('importScripts is unavailable in Node.js')
    }

    mediaPipeGlobal.self.import ??= async (specifier: string) => {
        const loaderPath = fileURLToPath(specifier)
        const source = await readFile(loaderPath, 'utf8')
        const module = { exports: {} as Record<string, unknown> }
        const require = createRequire(loaderPath)
        const evaluateLoader = new Function(
            'module',
            'exports',
            'require',
            '__dirname',
            '__filename',
            'globalThis',
            'self',
            `${source}\nreturn module.exports.default || module.exports || (typeof ModuleFactory !== "undefined" ? ModuleFactory : undefined);`,
        )
        mediaPipeGlobal.self!.ModuleFactory = evaluateLoader(
            module,
            module.exports,
            require,
            path.dirname(loaderPath),
            loaderPath,
            globalThis,
            mediaPipeGlobal.self,
        )
        return mediaPipeGlobal.self!.ModuleFactory
    }

    mediaPipeGlobal.OffscreenCanvas ??= class OffscreenCanvas {
        width: number
        height: number

        constructor (width: number, height: number) {
            this.width = width
            this.height = height
        }

        getContext (): null {
            return null
        }
    }
}

async function resolveDefaultWasmPath (): Promise<string> {
    return resolveMediaPipeWasmPath()
}

function detectLanguageWithMediaPipe (detector: LanguageDetectorInstance, sourceText: string): [string[], number, number] {
    if (!sourceText.trim()) {
        return [['no_text'], -1.0, -1.0]
    }

    if (HAS_UNUSUAL_ENG_REGEX.test(sourceText.trim())) {
        return [['un'], -1.0, -1.0]
    }

    const cleanedText = cleanTextForLanguageDetection(sourceText)

    if (isSymbolsOnly(cleanedText)) {
        return [['symbols_only'], -1.0, -1.0]
    }

    const noSymbolsText = removeSymbolsForLanguageDetection(cleanedText)

    if (!noSymbolsText) {
        return [['no_text'], -1.0, -1.0]
    }

    if (isSymbolsOnly(noSymbolsText)) {
        return [['symbols_only_again'], -1.0, -1.0]
    }

    if (HAS_UNUSUAL_ENG_REGEX.test(noSymbolsText)) {
        return [['un_again'], -1.0, -1.0]
    }

    let langResult = detector.detect(noSymbolsText).languages

    if (!langResult.length) {
        return [['un'], -1.0, -1.0]
    }

    let rawProb = langResult[0].probability
    let firstProb = rawProb
    let mediaPipeLangs = langResult.map(detection => detection.languageCode)

    if (mediaPipeLangs.some(lang => NON_LATIN_ISO_CODES.has(lang))) {
        const nonLatinText = noSymbolsText
            .replace(new RegExp(`[${VARIOUS_LETTERS_RANGE}'-]+`, 'g'), ' ')
            .replace(/\s+/g, ' ')
            .trim()

        if (nonLatinText) {
            const nonLatinLangResult = detector.detect(nonLatinText).languages

            if (nonLatinLangResult.length) {
                langResult = nonLatinLangResult
                rawProb = langResult[0].probability
                firstProb = rawProb
                mediaPipeLangs = langResult.map(detection => detection.languageCode)
            }
        }
    }

    if (langResult.length >= 2) {
        const secondProb = langResult[1].probability
        firstProb -= secondProb
        firstProb = adjustCloseLanguagePairConfidence(langResult[0].languageCode, langResult[1].languageCode, rawProb, secondProb, firstProb)
    }

    return [mediaPipeLangs, firstProb, rawProb]
}

function adjustCloseLanguagePairConfidence (firstLang: string, secondLang: string, rawProb: number, secondProb: number, defaultProb: number): number {
    if (firstLang === secondLang) {
        return defaultProb
    }

    const pair = new Set([firstLang, secondLang])

    if (!pair.has('id') || !pair.has('ms')) {
        return defaultProb
    }

    const weightedProb = rawProb - secondProb * 0.35
    const bonusProb = Math.min(rawProb + 0.15, 0.92)
    return Math.max(defaultProb, weightedProb, bonusProb)
}

function cleanTextForLanguageDetection (sourceText: string): string {
    let cleanedText = sourceText.replace(/\r\n|\r|\n/g, '__NEWLINE__').replaceAll('__NEWLINE__', ' ')
    cleanedText = cleanedText.replace(JS_VAR_PATTERN, '')
    cleanedText = replaceTagsWithValues(cleanedText)
    cleanedText = cleanedText.replace(XML_COLON_TAG_PATTERN, (_match, content: string) => tagHandler(content))
    cleanedText = removeHtmlTags(cleanedText).trim()
    cleanedText = cleanedText.replace(/^(\\n)+/g, '')
    cleanedText = cleanedText.replace(/(\\n)+$/g, '')
    cleanedText = cleanedText.replace(CLEAN_TEXT_PATTERN, ' ')
    return cleanedText.replace(/\s+/g, ' ').trim()
}

function replaceTagsWithValues (text: string): string {
    return text.replace(TAG_STYLE_PATTERN, match => {
        const values: string[] = []

        for (const valueMatch of match.matchAll(EXTRACT_TAG_ATTR_VALUE_PATTERN)) {
            values.push(valueMatch[2])
        }

        return ` ${values.join(' ')} `
    })
}

function tagHandler (content: string): string {
    const trimmed = content.trim()

    if (isDigitString(trimmed) || HAS_UNUSUAL_ENG_REGEX.test(trimmed)) {
        return ' '
    }

    return `${trimmed} `
}

function removeHtmlTags (sourceText: string): string {
    if (!sourceText) {
        return ''
    }

    const cleanedText = Array.from(sourceText)
        .filter(char => {
            const codePoint = char.codePointAt(0) ?? 0
            return codePoint < 0xD800 || codePoint > 0xDFFF
        })
        .join('')

    return DomUtils.textContent(parseDocument(cleanedText))
}

function isSymbolsOnly (sourceText: string): boolean {
    const cleanedText = sourceText.trim()

    if (!cleanedText) {
        return false
    }

    return Array.from(cleanedText).every(char => !/[\p{Letter}\p{Number}]/u.test(char))
}

function removeSymbolsForLanguageDetection (sourceText: string): string {
    let text = sourceText.replace(/[^\p{Letter}\p{Number}_\s「」『』，。、〜？,.'-]/gu, '')
    text = text.replace(/\d+/g, '')
    text = text.replace(/([\s「」『』，。、〜？,.-]+)/g, match => `${match[0]} `)
    text = text.replace(/\s+/g, ' ').trim()

    const noSymbolsAndNumText = Array.from(text)
        .filter(char => /[\p{Letter}\p{Number}]/u.test(char))
        .join('')

    if (Array.from(noSymbolsAndNumText).length <= 5) {
        return Array.from(text)
            .filter(char => /[\p{Letter}\p{Number}]/u.test(char) || /\s/u.test(char))
            .join('')
            .trim()
    }

    return text.trim()
}

async function buildKeyLanguageStats (root: string, items: KeyFilterItem[]): Promise<LanguageStats> {
    const stats = new Map<string, [number, number]>()
    let validItemsCount = 0
    const sourceTexts: string[] = []
    const yieldController = createYieldController()

    for (const [index, item] of items.entries()) {
        await yieldController.maybeYield(index)
        if (!item.lang_code) {
            continue
        }

        const [lang, confidence] = item.lang_code
        const value = stats.get(lang) ?? [0, 0.0]
        value[0] += 1
        value[1] += confidence
        stats.set(lang, value)
        validItemsCount += 1

        const finalDetectText = makeFinalDetectText(item)

        if (finalDetectText) {
            sourceTexts.push(finalDetectText)
        }
    }

    if (validItemsCount <= 0) {
        return [['un', 0, -1.0]]
    }

    const highThreshold = Math.max(validItemsCount * 0.1, Math.min(validItemsCount, 3))
    const midThreshold = Math.max(validItemsCount * 0.05, Math.min(validItemsCount, 2))
    const lowThreshold = Math.max(validItemsCount * 0.01, 1)
    const sortedLangs: LanguageStats = Array.from(stats.entries())
        .map(([lang, [count, totalConfidence]]) => [lang, count, totalConfidence / count] as [string, number, number])
        .sort((left, right) => right[1] - left[1] || right[2] - left[2])

    const highConfidenceLangs = sortedLangs.filter(([, count, avgConfidence]) =>
        (count >= highThreshold && avgConfidence >= 0.82) ||
        (count >= midThreshold && avgConfidence >= 0.92) ||
        (count >= lowThreshold && avgConfidence >= 0.96),
    )

    if (highConfidenceLangs.length) {
        return highConfidenceLangs
    }

    if (sourceTexts.length > 0) {
        const detector = await getLanguageDetector(root)
        const [langs, score] = detectLanguageWithMediaPipe(detector, sourceTexts.join('\n'))

        if (score >= 0.82) {
            return [[langs[0], sourceTexts.length, score]]
        }

        return [['un', sourceTexts.length, -1.0]]
    }

    return [['un', 0, -1.0]]
}

function buildLowConfidenceKeyLanguageStats (items: KeyFilterItem[], languageStats: LanguageStats): LanguageStats {
    const highConfidenceSet = new Set(languageStats.map(([lang]) => lang))
    const stats = new Map<string, [number, number]>()

    for (const item of items) {
        if (!item.lang_code) {
            continue
        }

        const [lang, confidence] = item.lang_code
        const value = stats.get(lang) ?? [0, 0.0]
        value[0] += 1
        value[1] += confidence
        stats.set(lang, value)
    }

    return Array.from(stats.entries())
        .map(([lang, [count, totalConfidence]]) => [lang, count, totalConfidence / count] as [string, number, number])
        .filter(([lang]) => !highConfidenceSet.has(lang))
        .sort((left, right) => right[1] - left[1] || right[2] - left[2])
}

function makeFinalDetectText (item: KeyFilterItem): string {
    const noSymbolsText = removeSymbolsForLanguageDetection(cleanTextForLanguageDetection(item.source_text))
    const langs = new Set([item.lang_code?.[0] ?? '', ...(item.lang_code?.[2] ?? [])])

    if (Array.from(langs).some(lang => NON_LATIN_ISO_CODES.has(lang))) {
        return noSymbolsText
            .replace(new RegExp(`[${VARIOUS_LETTERS_RANGE}'-]+`, 'g'), ' ')
            .replace(/\s+/g, ' ')
            .trim()
    }

    return noSymbolsText
}

function filterKeyItemsByLanguage (items: KeyFilterItem[], context: {
    sourceLanguage: string
    targetLanguage: string
    languageStats: LanguageStats
    lcLanguageStats: LanguageStats
}): void {
    const target: KeyFilterItem[] = []

    if (context.sourceLanguage === 'auto') {
        const mostCommonLanguage = getMostCommonLanguage(context.languageStats)
        const firstLanguage = context.languageStats.length ? context.languageStats[0][0] : mostCommonLanguage

        if (mapLanguageCodeToName(firstLanguage) === context.targetLanguage) {
            target.push(...filterTargetLanguageMatch(items, firstLanguage))
        } else if (firstLanguage === 'un') {
            target.push(...filterUnknownLanguage(items))
        } else {
            target.push(...filterNormalLanguage(items, firstLanguage, context))
        }
    } else {
        target.push(...filterNormalLanguage(items, context.sourceLanguage, context))
    }

    for (const item of target) {
        item.translation_status = translationStatus.EXCLUDED
    }
}

function getMostCommonLanguage (languageStats: LanguageStats): string {
    const counts = new Map<string, number>()

    for (const [lang, count] of languageStats) {
        if (lang === 'un') {
            continue
        }

        counts.set(lang, (counts.get(lang) ?? 0) + count)
    }

    if (!counts.size) {
        return 'un'
    }

    return Array.from(counts.entries()).sort((left, right) => right[1] - left[1])[0][0]
}

function filterTargetLanguageMatch (items: KeyFilterItem[], language: string): KeyFilterItem[] {
    const hasAny = getLanguageFilterFunction(language)

    if (hasAny) {
        return items.filter(item =>
            item.translation_status === translationStatus.EXCLUDED ||
            !item.lang_code ||
            (hasAny(item.source_text) && item.lang_code[0] === language && item.lang_code[1] > 0.92),
        )
    }

    return items.filter(item =>
        item.translation_status === translationStatus.EXCLUDED ||
        !item.lang_code ||
        (item.lang_code[0] === language && item.lang_code[1] > 0.92),
    )
}

function filterUnknownLanguage (items: KeyFilterItem[]): KeyFilterItem[] {
    return items.filter(item =>
        item.translation_status === translationStatus.EXCLUDED ||
        !item.lang_code ||
        item.lang_code[1] < 0.82,
    )
}

function filterNormalLanguage (items: KeyFilterItem[], language: string, context: { lcLanguageStats?: LanguageStats } = {}): KeyFilterItem[] {
    let mainSourceLang = mapLanguageNameToCode(language)

    if (mainSourceLang === 'zh-Hant') {
        mainSourceLang = 'zh'
    }

    const hasAny = getLanguageFilterFunction(mainSourceLang)
    const lcLanguages = new Set((context.lcLanguageStats ?? []).map(([lang]) => lang))
    const filteredItems: KeyFilterItem[] = []

    for (const item of items) {
        if (item.translation_status === translationStatus.EXCLUDED) {
            filteredItems.push(item)
            continue
        }

        const langInfo = item.lang_code ?? [mainSourceLang, 1.0, []]
        const [detectedLang, confidence, otherLangs = []] = langInfo
        const notFilterForLc = detectedLang !== mainSourceLang && lcLanguages.has(detectedLang) && otherLangs.includes(mainSourceLang)

        if (notFilterForLc) {
            continue
        }

        if (hasAny) {
            if (!hasAny(item.source_text) || (detectedLang !== mainSourceLang && confidence > 0.92)) {
                filteredItems.push(item)
            }
        } else if (detectedLang !== mainSourceLang && confidence > 0.92) {
            filteredItems.push(item)
        }
    }

    return filteredItems
}

function getLanguageFilterFunction (languageCode: string): ((text: string) => boolean)|null {
    const code = languageCode && LANGUAGE_FILTERS[languageCode] ? languageCode : languageCode?.slice(0, 2)
    return LANGUAGE_FILTERS[code] ?? null
}

const LANGUAGE_FILTERS: Record<string, (text: string) => boolean> = Object.freeze({
    zh: hasAnyCjk,
    'zh-cn': hasAnyCjk,
    'zh-tw': hasAnyCjk,
    yue: hasAnyCjk,
    en: hasAnyLatinScript,
    es: hasAnyLatinScript,
    fr: hasAnyLatinScript,
    de: hasAnyLatinScript,
    id: hasAnyLatinScript,
    in: hasAnyLatinScript,
    vi: hasAnyLatinScript,
    ko: hasAnyKorean,
    ru: hasAnyRussian,
    th: hasAnyThai,
    ja: hasAnyJapanese,
})

function hasAnyCjk (text: string): boolean {
    return Array.from(text).some(isFilterCjk)
}

function hasAnyLatinScript (text: string): boolean {
    return Array.from(text).some(isFilterLatin)
}

function hasAnyKorean (text: string): boolean {
    return Array.from(text).some(isFilterKorean)
}

function hasAnyRussian (text: string): boolean {
    return Array.from(text).some(isFilterRussian)
}

function hasAnyThai (text: string): boolean {
    return Array.from(text).some(isFilterThai)
}

function hasAnyJapanese (text: string): boolean {
    return Array.from(text).some(isFilterJapanese)
}

function isFilterCjk (char: string): boolean {
    return inCodePointRange(char, 0x4E00, 0x9FFF)
}

function isFilterLatin (char: string): boolean {
    return (
        inCodePointRange(char, 0x0041, 0x005A) ||
        inCodePointRange(char, 0x0061, 0x007A) ||
        inCodePointRange(char, 0x0100, 0x017F) ||
        inCodePointRange(char, 0x0180, 0x024F) ||
        inCodePointRange(char, 0x00A0, 0x00FF)
    )
}

function isFilterKorean (char: string): boolean {
    return (
        isFilterCjk(char) ||
        inCodePointRange(char, 0x1100, 0x11FF) ||
        inCodePointRange(char, 0xA960, 0xA97F) ||
        inCodePointRange(char, 0xD7B0, 0xD7FF) ||
        inCodePointRange(char, 0xAC00, 0xD7AF) ||
        inCodePointRange(char, 0x3130, 0x318F)
    )
}

function isFilterRussian (char: string): boolean {
    return (
        inCodePointRange(char, 0x0410, 0x044F) ||
        inCodePointRange(char, 0x0500, 0x052F) ||
        inCodePointRange(char, 0x2C00, 0x2C5F) ||
        inCodePointRange(char, 0x0300, 0x04FF) ||
        inCodePointRange(char, 0x1C80, 0x1C8F) ||
        inCodePointRange(char, 0x2DE0, 0x2DFF) ||
        inCodePointRange(char, 0x0500, 0x050F)
    )
}

function isFilterThai (char: string): boolean {
    return inCodePointRange(char, 0x0E00, 0x0E7F)
}

function isFilterJapanese (char: string): boolean {
    return (
        isFilterCjk(char) ||
        inCodePointRange(char, 0x30A0, 0x30FF) ||
        inCodePointRange(char, 0x3040, 0x309F) ||
        inCodePointRange(char, 0xFF65, 0xFF9F) ||
        inCodePointRange(char, 0x31F0, 0x31FF) ||
        inCodePointRange(char, 0x309B, 0x309C)
    )
}

function inCodePointRange (char: string, start: number, end: number): boolean {
    const codePoint = char.codePointAt(0) ?? 0
    return codePoint >= start && codePoint <= end
}

function mapLanguageNameToCode (languageName: string): string {
    return LANGUAGE_NAME_TO_CODE[languageName] ?? languageName
}

function mapLanguageCodeToName (languageCode: string): string {
    return LANGUAGE_CODE_TO_NAME[languageCode] ?? languageCode
}

function isDigitString (text: string): boolean {
    return /^\p{Number}+$/u.test(text)
}

export function isGarbledOrBinaryLikeText (text: string): boolean {
    const trimmed = text.trim()

    if (!trimmed) {
        return false
    }

    if (hasZlibLikeHeader(trimmed) || trimmed.includes('\uFFFD') || MOJIBAKE_SIGNAL_PATTERN.test(trimmed)) {
        return true
    }

    return Array.from(trimmed).some(isBinaryControlChar)
}

function hasZlibLikeHeader (text: string): boolean {
    if (text.length < 2 || text[0] !== 'x') {
        return false
    }

    const flagByte = text.codePointAt(1) ?? 0
    return flagByte === 0x01 || flagByte === 0x5E || flagByte === 0x9C || flagByte === 0xDA
}

function isBinaryControlChar (char: string): boolean {
    const codePoint = char.codePointAt(0) ?? 0
    return (
        codePoint === 0x00 ||
        (codePoint < 0x20 && codePoint !== 0x09 && codePoint !== 0x0A && codePoint !== 0x0D) ||
        (codePoint >= 0x7F && codePoint <= 0x9F)
    )
}

function isPunctuationString (text: string): boolean {
    return Array.from(text).every(char => PUNCTUATION_CHARS.has(char))
}

function getFileSuffix (text: string): string {
    const dotIndex = text.lastIndexOf('.')
    return dotIndex === -1 ? text : text.slice(dotIndex)
}

function escapeRegex (text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
