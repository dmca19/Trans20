import { createReadStream } from 'node:fs'
import { open, readdir, readFile, stat } from 'node:fs/promises'
import path from 'node:path'

import { tool, type StructuredToolInterface } from '@langchain/core/tools'
import kuromoji, { type IpadicFeatures, type Tokenizer } from 'kuromoji'
// LangChain tool schemas intentionally use Zod v3; do not switch this to the package default v4 import.
import { z } from 'zod/v3'

import { createYieldController } from './eventLoop.js'
import { parseJsonText } from './fileUtils.js'
import {
    filterManualTransKeyItems,
    normalizeKeyFilterOptions,
    type FilteredManualTransKey,
    type KeyFilter,
    type KeyFilterOptions,
} from './keyFilter.js'
import { resolveProjectFile as resolveSafeProjectFile, resolveProjectPath, toPosixPath } from './pathUtils.js'
import { resolveKuromojiDictionaryPath } from './runtimeResources.js'
import { truncate, withToolLogging, type ToolCallEvent, type ToolCallLogger } from './toolRuntime.js'

const MAX_READ_LINES = 200
const MAX_SEARCH_RESULTS = 50
const MAX_SEARCH_FILES = 2000
const MAX_SEARCH_FILE_BYTES = 1_000_000
const MAX_TOOL_OUTPUT_CHARS = 12000
const BINARY_SAMPLE_BYTES = 8192
const DEFAULT_KEY_LIMIT = 100
const MAX_KEY_LIMIT = 500
const DEFAULT_KEY_CONTEXT = 5
const MAX_KEY_CONTEXT = 50
const DEFAULT_KEY_EXAMPLES = 5
const MAX_KEY_EXAMPLES = 20
const TOP_FREQUENCY_TERMS = 30
const TOP_FREQUENCY_PHRASES = 20
const MIN_HIGH_FREQUENCY_TERM_COUNT = 5
const MIN_REPEATED_PHRASE_COUNT = 3

const KUROMOJI_DIC_PATH = resolveKuromojiDictionaryPath()
const kuromojiRuntime = loadKuromojiRuntime()

let kuromojiTokenizerPromise: Promise<Tokenizer<IpadicFeatures>>|null = null

const skippedDirectories = new Set([
    '.git',
    '.idea',
    '.vscode',
    'node_modules',
    'dist',
    'build',
    'coverage',
])

const skippedExtensions = new Set([
    '.7z',
    '.avif',
    '.bmp',
    '.class',
    '.dll',
    '.exe',
    '.gif',
    '.ico',
    '.jar',
    '.jpg',
    '.jpeg',
    '.lockb',
    '.pdf',
    '.png',
    '.so',
    '.webp',
    '.zip',
])

const sensitiveRootFiles = new Set([
    'config.json',
])

export type { ToolCallEvent, ToolCallLogger }

type KeyFileCacheEntry = {
    realFilePath: string
    size: number
    mtimeMs: number
    keys: string[]
}

type KeyFilterCacheEntry = {
    realFilePath: string
    size: number
    mtimeMs: number
    filterLanguage: string
    keyItems: FilteredManualTransKey[]
}

type KeySet = {
    relativePath: string
    realFilePath: string
    size: number
    mtimeMs: number
    originalKeys: string[]
    keyItems: FilteredManualTransKey[]
    keys: string[]
    filter: KeyFilter
    scope: KeyInspectionScope|null
}

export type KeyInspectionScope = {
    batch_start_index: number
    batch_end_index: number
    context_window: number
    allowed_key_indices?: number[]
    key_index_to_filtered_index?: Record<number, number>
    filtered_index_to_key_index?: Record<number, number>
}

export type ManualTransKeyItem = FilteredManualTransKey

export type CreateInspectionToolsOptions = {
    keyScope?: KeyInspectionScope
    keyOnly?: boolean
    exposeRawKeyIndex?: boolean
    allowUnfiltered?: boolean
}

type KuromojiRuntime = {
    builder: (option: { dicPath: string }) => {
        build: (callback: (error: Error|null, tokenizer: Tokenizer<IpadicFeatures>) => void) => void
    }
}

const keyFileCache = new Map<string, KeyFileCacheEntry>()
const keyFilterCache = new Map<string, KeyFilterCacheEntry>()

const keyFilterSchema = {
    filter: z.boolean().optional().describe('Filter keys before returning or analyzing them. Defaults to true.'),
    filterLanguage: z.string().trim().min(1).optional().describe('Source language used by the key filter. Defaults to auto.'),
}

export function createInspectionTools (
    root: string,
    onToolEvent: ToolCallLogger,
    manualTransFile: string,
    options: CreateInspectionToolsOptions = {},
): StructuredToolInterface[] {
    const projectFileTools = [
        tool(async input => withToolLogging('get_file_info', input, onToolEvent, async () => {
            const file = await resolveProjectFile(root, input.filePath)
            const extension = path.extname(file.realFilePath).toLowerCase()
            const binaryByExtension = skippedExtensions.has(extension)
            const binaryByContent = binaryByExtension ? false : await fileContainsNullByte(file.realFilePath)
            const isBinary = binaryByExtension || binaryByContent
            const lineCount = isBinary ? null : await countTextFileLines(file.realFilePath, file.fileStat.size)
            const searchable = !isBinary && isSearchableFile(root, file.realFilePath) && file.fileStat.size <= MAX_SEARCH_FILE_BYTES
            const readable = !isBinary

            return [
                `Path: ${file.normalizedRelativePath}`,
                `Size: ${file.fileStat.size} bytes`,
                `Lines: ${lineCount ?? 'unknown'}`,
                `Extension: ${extension || '(none)'}`,
                `Binary: ${isBinary}`,
                `Readable by line tools: ${readable}`,
                `Searchable by search tools: ${searchable}`,
                `Suggested next step: ${suggestNextStep(isBinary, file.fileStat.size, lineCount)}`,
            ].join('\n')
        }), {
            name: 'get_file_info',
            description: 'Get size, line count, binary status, and suggested next inspection step for one project file.',
            schema: z.object({
                filePath: z.string().describe('Project-relative file path to inspect.'),
            }),
        }),
        tool(async input => withToolLogging('read_file_lines', input, onToolEvent, async () => {
            const text = await readProjectTextFile(root, input.filePath)
            const lines = splitLines(text)
            const startLine = input.startLine
            const endLine = Math.min(input.endLine ?? startLine + MAX_READ_LINES - 1, lines.length)
            const safeEndLine = Math.min(endLine, startLine + MAX_READ_LINES - 1)

            if (startLine > lines.length) {
                return `File has ${lines.length} lines. startLine ${startLine} is outside the file.`
            }

            return limitToolOutput(formatLineRange(lines, startLine, safeEndLine))
        }), {
            name: 'read_file_lines',
            description: `Read a 1-based line range from a project text file. Returns at most ${MAX_READ_LINES} lines.`,
            schema: z.object({
                filePath: z.string().describe('Project-relative path to read.'),
                startLine: z.number().int().min(1).describe('1-based first line to read.'),
                endLine: z.number().int().min(1).optional().describe('1-based last line to read. Defaults to a short range.'),
            }),
        }),
        tool(async input => withToolLogging('search_in_files', input, onToolEvent, async () => {
            const matcher = createMatcher(input.pattern, input.isRegex ?? false, input.caseSensitive ?? false)
            const files = await listSearchableFiles(root)
            const maxResults = Math.min(input.maxResults ?? 20, MAX_SEARCH_RESULTS)
            const results: string[] = []
            const yieldController = createYieldController()

            for (const [fileIndex, filePath] of files.entries()) {
                await yieldController.maybeYield(fileIndex)
                const text = await readFile(filePath, 'utf8')
                const lines = splitLines(text)
                const relativePath = toProjectPath(root, filePath)

                for (let index = 0; index < lines.length; index += 1) {
                    await yieldController.maybeYield(index)
                    if (!matcher.test(lines[index])) {
                        continue
                    }

                    results.push(`${relativePath}:${index + 1}: ${lines[index].trim()}`)

                    if (results.length >= maxResults) {
                        return limitToolOutput([
                            `Found at least ${results.length} matches. Results are truncated.`,
                            ...results,
                        ].join('\n'))
                    }
                }
            }

            if (results.length === 0) {
                return 'No matches found.'
            }

            return limitToolOutput([
                `Found ${results.length} matches.`,
                ...results,
            ].join('\n'))
        }), {
            name: 'search_in_files',
            description: 'Search project text files for a literal string or regular expression.',
            schema: z.object({
                pattern: z.string().min(1).describe('Text or regex pattern to search for.'),
                isRegex: z.boolean().optional().describe('Treat pattern as a JavaScript regular expression. Defaults to false.'),
                caseSensitive: z.boolean().optional().describe('Use case-sensitive matching. Defaults to false.'),
                maxResults: z.number().int().min(1).max(MAX_SEARCH_RESULTS).optional().describe('Maximum matches to return. Defaults to 20.'),
            }),
        }),
        tool(async input => withToolLogging('sample_file_lines', input, onToolEvent, async () => {
            const text = await readProjectTextFile(root, input.filePath)
            const lines = splitLines(text)
            const sectionSize = Math.min(input.linesPerSection ?? 20, 50)

            if (lines.length <= sectionSize * 3) {
                return limitToolOutput(formatLineRange(lines, 1, lines.length))
            }

            const middleStart = Math.max(1, Math.floor(lines.length / 2) - Math.floor(sectionSize / 2))
            const sections = [
                `--- head: lines 1-${sectionSize} ---`,
                formatLineRange(lines, 1, sectionSize),
                `--- middle: lines ${middleStart}-${middleStart + sectionSize - 1} ---`,
                formatLineRange(lines, middleStart, middleStart + sectionSize - 1),
                `--- tail: lines ${lines.length - sectionSize + 1}-${lines.length} ---`,
                formatLineRange(lines, lines.length - sectionSize + 1, lines.length),
            ]

            return limitToolOutput(sections.join('\n'))
        }), {
            name: 'sample_file_lines',
            description: 'Sample head, middle, and tail lines from a project text file.',
            schema: z.object({
                filePath: z.string().describe('Project-relative path to sample.'),
                linesPerSection: z.number().int().min(1).max(50).optional().describe('Lines per head/middle/tail section. Defaults to 20.'),
            }),
        }),
        tool(async input => withToolLogging('search_patterns_summary', input, onToolEvent, async () => {
            const files = await listSearchableFiles(root)
            const maxExamples = Math.min(input.maxExamplesPerPattern ?? 3, 10)
            const summaries: string[] = []

            for (const pattern of input.patterns) {
                const matcher = createMatcher(pattern, input.isRegex ?? false, input.caseSensitive ?? false)
                let matchingFiles = 0
                let matchCount = 0
                const examples: string[] = []
                const yieldController = createYieldController()

                for (const [fileIndex, filePath] of files.entries()) {
                    await yieldController.maybeYield(fileIndex)
                    const text = await readFile(filePath, 'utf8')
                    const lines = splitLines(text)
                    let fileMatched = false

                    for (let index = 0; index < lines.length; index += 1) {
                        await yieldController.maybeYield(index)
                        if (!matcher.test(lines[index])) {
                            continue
                        }

                        fileMatched = true
                        matchCount += 1

                        if (examples.length < maxExamples) {
                            examples.push(`${toProjectPath(root, filePath)}:${index + 1}: ${lines[index].trim()}`)
                        }
                    }

                    if (fileMatched) {
                        matchingFiles += 1
                    }
                }

                summaries.push([
                    `Pattern: ${pattern}`,
                    `Matching files: ${matchingFiles}`,
                    `Matching lines: ${matchCount}`,
                    examples.length > 0 ? `Examples:\n${examples.map(example => `  ${example}`).join('\n')}` : 'Examples: none',
                ].join('\n'))
            }

            return limitToolOutput(summaries.join('\n\n'))
        }), {
            name: 'search_patterns_summary',
            description: 'Summarize project-wide matches for multiple literal strings or regular expressions.',
            schema: z.object({
                patterns: z.array(z.string().min(1)).min(1).max(20).describe('Patterns to summarize.'),
                isRegex: z.boolean().optional().describe('Treat patterns as JavaScript regular expressions. Defaults to false.'),
                caseSensitive: z.boolean().optional().describe('Use case-sensitive matching. Defaults to false.'),
                maxExamplesPerPattern: z.number().int().min(0).max(10).optional().describe('Maximum examples per pattern. Defaults to 3.'),
            }),
        }),
        tool(async input => withToolLogging('get_file_excerpt_by_match', input, onToolEvent, async () => {
            const text = await readProjectTextFile(root, input.filePath)
            const lines = splitLines(text)
            const matcher = createMatcher(input.pattern, input.isRegex ?? false, input.caseSensitive ?? false)
            const matchIndex = input.matchIndex ?? 1
            let seenMatches = 0

            for (let index = 0; index < lines.length; index += 1) {
                if (!matcher.test(lines[index])) {
                    continue
                }

                seenMatches += 1

                if (seenMatches !== matchIndex) {
                    continue
                }

                const beforeLines = Math.min(input.beforeLines ?? 3, 20)
                const afterLines = Math.min(input.afterLines ?? 3, 20)
                const startLine = Math.max(1, index + 1 - beforeLines)
                const endLine = Math.min(lines.length, index + 1 + afterLines)

                return limitToolOutput(formatLineRange(lines, startLine, endLine))
            }

            return `No match ${matchIndex} found for pattern "${input.pattern}".`
        }), {
            name: 'get_file_excerpt_by_match',
            description: 'Return nearby lines around the Nth match of a pattern in one project text file.',
            schema: z.object({
                filePath: z.string().describe('Project-relative file path to inspect.'),
                pattern: z.string().min(1).describe('Text or regex pattern to find.'),
                isRegex: z.boolean().optional().describe('Treat pattern as a JavaScript regular expression. Defaults to false.'),
                caseSensitive: z.boolean().optional().describe('Use case-sensitive matching. Defaults to false.'),
                beforeLines: z.number().int().min(0).max(20).optional().describe('Context lines before the match. Defaults to 3.'),
                afterLines: z.number().int().min(0).max(20).optional().describe('Context lines after the match. Defaults to 3.'),
                matchIndex: z.number().int().min(1).optional().describe('1-based match occurrence to inspect. Defaults to 1.'),
            }),
        }),
    ]

    const keyTools = [
        tool(async input => withToolLogging('get_key_file_info', input, onToolEvent, async () => {
            const keySet = await loadManualTransKeySet(root, manualTransFile, normalizeToolKeyFilterInput(input, options), options.keyScope)
            const sampleSize = Math.min(input.sampleSize ?? 5, 20)
            const firstKeyItems = keySet.keyItems.slice(0, sampleSize)
            const lastKeyItems = sampleSize === 0 ? [] : keySet.keyItems.slice(-sampleSize)

            return limitToolOutput([
                ...formatKeySetHeader(keySet),
                'JSON type: object',
                `Size: ${keySet.size} bytes`,
                `First keys:\n${formatKeyItemList(firstKeyItems, options)}`,
                `Last keys:\n${formatKeyItemList(lastKeyItems, options)}`,
                'Suggested next step: use read_key_range with index ranges or search_keys for targeted lookup.',
            ].join('\n'))
        }), {
            name: 'get_key_file_info',
            description: 'Get key count, file size, and key samples from the configured ManualTrans JSON file. Values are never returned.',
            schema: z.object({
                sampleSize: z.number().int().min(0).max(20).optional().describe('Number of first and last keys to show. Defaults to 5.'),
                ...keyFilterSchema,
            }),
        }),
        tool(async input => withToolLogging('inspect_json_kv_file', input, onToolEvent, async () => {
            const sampleLimit = Math.min(input.sampleLimit ?? 10, 50)
            return inspectJsonKvFile(root, manualTransFile, sampleLimit)
        }), {
            name: 'inspect_json_kv_file',
            description: 'Inspect the configured ManualTrans JSON file as a JSON key-value translation object. Values are summarized, not returned.',
            schema: z.object({
                sampleLimit: z.number().int().min(0).max(50).optional().describe('Maximum anomaly samples per category. Defaults to 10.'),
            }),
        }),
        tool(async input => withToolLogging('analyze_key_language_mix', input, onToolEvent, async () => {
            const keySet = await loadManualTransKeySet(root, manualTransFile, normalizeToolKeyFilterInput(input, options), options.keyScope)
            const result = withKeyFilterMetadata(await analyzeKeyLanguageMix(keySet.relativePath, keySet.keys), 'key_language_mix', keySet)
            return limitToolOutput(JSON.stringify(result, null, 2))
        }), {
            name: 'analyze_key_language_mix',
            description: 'Analyze language, script, and translation-protection hints for ManualTrans JSON object keys. Values are never returned.',
            schema: z.object({
                ...keyFilterSchema,
            }),
        }),
        tool(async input => withToolLogging('analyze_key_frequency_profile', input, onToolEvent, async () => {
            const keySet = await loadManualTransKeySet(root, manualTransFile, normalizeToolKeyFilterInput(input, options), options.keyScope)
            const tokenizer = await getKuromojiTokenizer()
            const result = withKeyFilterMetadata(await analyzeKeyFrequencyProfile(keySet.relativePath, keySet.keys, tokenizer), 'frequency_profile', keySet)
            return limitToolOutput(JSON.stringify(result, null, 2))
        }), {
            name: 'analyze_key_frequency_profile',
            description: 'Use kuromoji to clean ManualTrans JSON keys, tokenize Japanese text, and summarize term/phrase frequency for translation preparation. Values are never returned.',
            schema: z.object({
                ...keyFilterSchema,
            }),
        }),
        tool(async input => withToolLogging('read_key_range', input, onToolEvent, async () => {
            const keySet = await loadManualTransKeySet(root, manualTransFile, normalizeToolKeyFilterInput(input, options), options.keyScope)
            const startIndex = input.startIndex ?? keySet.keyItems[0]?.filteredIndex ?? 0
            const requestedEndIndex = input.endIndex ?? startIndex + DEFAULT_KEY_LIMIT - 1
            const endIndex = Math.min(requestedEndIndex, startIndex + MAX_KEY_LIMIT - 1)
            const selectedKeyItems = selectKeyItemsByFilteredRange(keySet.keyItems, startIndex, endIndex)

            return limitToolOutput([
                ...formatKeySetHeader(keySet),
                `Start index: ${startIndex}`,
                `End index: ${endIndex}`,
                `Returned: ${selectedKeyItems.length}`,
                `Keys:\n${formatIndexedKeyList(selectedKeyItems, options)}`,
            ].join('\n'))
        }), {
            name: 'read_key_range',
            description: 'Read a 0-based index range of ManualTrans JSON object keys. Values are never returned.',
            schema: z.object({
                startIndex: z.number().int().min(0).optional().describe('0-based first key index to read. Defaults to 0.'),
                endIndex: z.number().int().min(0).optional().describe(`0-based last key index to read. Defaults to startIndex + ${DEFAULT_KEY_LIMIT - 1}; capped to ${MAX_KEY_LIMIT} returned keys.`),
                ...keyFilterSchema,
            }),
        }),
        tool(async input => withToolLogging('search_keys', input, onToolEvent, async () => {
            const keySet = await loadManualTransKeySet(root, manualTransFile, normalizeToolKeyFilterInput(input, options), options.keyScope)
            const matcher = createMatcher(input.pattern, input.isRegex ?? false, input.caseSensitive ?? false)
            const limit = Math.min(input.limit ?? DEFAULT_KEY_LIMIT, MAX_KEY_LIMIT)
            const matchedKeyItems: FilteredManualTransKey[] = []
            let totalMatched = 0
            const yieldController = createYieldController()

            for (const [index, keyItem] of keySet.keyItems.entries()) {
                await yieldController.maybeYield(index)
                if (!matcher.test(keyItem.key)) {
                    continue
                }

                totalMatched += 1

                if (matchedKeyItems.length < limit) {
                    matchedKeyItems.push(keyItem)
                }
            }

            return limitToolOutput([
                ...formatKeySetHeader(keySet),
                `Pattern: ${input.pattern}`,
                `Total matched keys: ${totalMatched}`,
                `Returned: ${matchedKeyItems.length}`,
                `Keys:\n${formatKeyItemList(matchedKeyItems, options)}`,
            ].join('\n'))
        }), {
            name: 'search_keys',
            description: 'Search keys in the configured ManualTrans JSON file without returning values.',
            schema: z.object({
                pattern: z.string().min(1).describe('Text or regex pattern to search keys for.'),
                isRegex: z.boolean().optional().describe('Treat pattern as a JavaScript regular expression. Defaults to false.'),
                caseSensitive: z.boolean().optional().describe('Use case-sensitive matching. Defaults to false.'),
                limit: z.number().int().min(1).max(MAX_KEY_LIMIT).optional().describe(`Maximum keys to return. Defaults to ${DEFAULT_KEY_LIMIT}, max ${MAX_KEY_LIMIT}.`),
                ...keyFilterSchema,
            }),
        }),
        tool(async input => withToolLogging('sample_keys', input, onToolEvent, async () => {
            const keySet = await loadManualTransKeySet(root, manualTransFile, normalizeToolKeyFilterInput(input, options), options.keyScope)
            const sectionSize = Math.min(input.keysPerSection ?? 20, 100)

            if (keySet.keys.length <= sectionSize * 3) {
                return limitToolOutput([
                    ...formatKeySetHeader(keySet),
                    `Keys:\n${formatKeyItemList(keySet.keyItems, options)}`,
                ].join('\n'))
            }

            const middleStart = Math.max(0, Math.floor(keySet.keys.length / 2) - Math.floor(sectionSize / 2))
            const sections = [
                ...formatKeySetHeader(keySet),
                `--- head: indexes 0-${sectionSize - 1} ---`,
                formatIndexedKeyList(keySet.keyItems.slice(0, sectionSize), options),
                `--- middle: indexes ${middleStart}-${middleStart + sectionSize - 1} ---`,
                formatIndexedKeyList(keySet.keyItems.slice(middleStart, middleStart + sectionSize), options),
                `--- tail: indexes ${keySet.keys.length - sectionSize}-${keySet.keys.length - 1} ---`,
                formatIndexedKeyList(keySet.keyItems.slice(keySet.keys.length - sectionSize, keySet.keys.length), options),
            ]

            return limitToolOutput(sections.join('\n'))
        }), {
            name: 'sample_keys',
            description: 'Sample head, middle, and tail keys from the configured ManualTrans JSON file. Values are never returned.',
            schema: z.object({
                keysPerSection: z.number().int().min(1).max(100).optional().describe('Keys per head/middle/tail section. Defaults to 20.'),
                ...keyFilterSchema,
            }),
        }),
        tool(async input => withToolLogging('summarize_key_patterns', input, onToolEvent, async () => {
            const keySet = await loadManualTransKeySet(root, manualTransFile, normalizeToolKeyFilterInput(input, options), options.keyScope)
            const maxExamples = Math.min(input.maxExamplesPerPattern ?? DEFAULT_KEY_EXAMPLES, MAX_KEY_EXAMPLES)
            const summaries: string[] = []

            for (const pattern of input.patterns) {
                const matcher = createMatcher(pattern, input.isRegex ?? false, input.caseSensitive ?? false)
                let matchCount = 0
                const examples: FilteredManualTransKey[] = []
                const yieldController = createYieldController()

                for (const [index, keyItem] of keySet.keyItems.entries()) {
                    await yieldController.maybeYield(index)
                    if (!matcher.test(keyItem.key)) {
                        continue
                    }

                    matchCount += 1

                    if (examples.length < maxExamples) {
                        examples.push(keyItem)
                    }
                }

                summaries.push([
                    `Pattern: ${pattern}`,
                    `Matching keys: ${matchCount}`,
                    examples.length > 0 ? `Examples:\n${formatKeyItemList(examples, options)}` : 'Examples: none',
                ].join('\n'))
            }

            return limitToolOutput([
                ...formatKeySetHeader(keySet),
                '',
                summaries.join('\n\n'),
            ].join('\n'))
        }), {
            name: 'summarize_key_patterns',
            description: 'Summarize key matches for multiple patterns in the configured ManualTrans JSON file.',
            schema: z.object({
                patterns: z.array(z.string().min(1)).min(1).max(20).describe('Patterns to summarize against keys.'),
                isRegex: z.boolean().optional().describe('Treat patterns as JavaScript regular expressions. Defaults to false.'),
                caseSensitive: z.boolean().optional().describe('Use case-sensitive matching. Defaults to false.'),
                maxExamplesPerPattern: z.number().int().min(0).max(MAX_KEY_EXAMPLES).optional().describe(`Maximum key examples per pattern. Defaults to ${DEFAULT_KEY_EXAMPLES}.`),
                ...keyFilterSchema,
            }),
        }),
        tool(async input => withToolLogging('get_key_excerpt_by_match', input, onToolEvent, async () => {
            const keySet = await loadManualTransKeySet(root, manualTransFile, normalizeToolKeyFilterInput(input, options), options.keyScope)
            const matcher = createMatcher(input.pattern, input.isRegex ?? false, input.caseSensitive ?? false)
            const matchIndex = input.matchIndex ?? 1
            let seenMatches = 0

            for (let index = 0; index < keySet.keys.length; index += 1) {
                if (!matcher.test(keySet.keys[index])) {
                    continue
                }

                seenMatches += 1

                if (seenMatches !== matchIndex) {
                    continue
                }

                const beforeKeys = Math.min(input.beforeKeys ?? DEFAULT_KEY_CONTEXT, MAX_KEY_CONTEXT)
                const afterKeys = Math.min(input.afterKeys ?? DEFAULT_KEY_CONTEXT, MAX_KEY_CONTEXT)
                const startIndex = Math.max(0, index - beforeKeys)
                const endIndex = Math.min(keySet.keys.length, index + afterKeys + 1)

                return limitToolOutput([
                    ...formatKeySetHeader(keySet),
                    `Pattern: ${input.pattern}`,
                    `Match index: ${matchIndex}`,
                    `Matched filtered_key_index: ${keySet.keyItems[index]?.filteredIndex ?? index}`,
                    `Keys:\n${formatIndexedKeyList(keySet.keyItems.slice(startIndex, endIndex), options)}`,
                ].join('\n'))
            }

            return `No key match ${matchIndex} found for pattern "${input.pattern}".`
        }), {
            name: 'get_key_excerpt_by_match',
            description: 'Return nearby keys around the Nth matching key in the configured ManualTrans JSON file. Values are never returned.',
            schema: z.object({
                pattern: z.string().min(1).describe('Text or regex pattern to find in keys.'),
                isRegex: z.boolean().optional().describe('Treat pattern as a JavaScript regular expression. Defaults to false.'),
                caseSensitive: z.boolean().optional().describe('Use case-sensitive matching. Defaults to false.'),
                beforeKeys: z.number().int().min(0).max(MAX_KEY_CONTEXT).optional().describe(`Keys before the match. Defaults to ${DEFAULT_KEY_CONTEXT}.`),
                afterKeys: z.number().int().min(0).max(MAX_KEY_CONTEXT).optional().describe(`Keys after the match. Defaults to ${DEFAULT_KEY_CONTEXT}.`),
                matchIndex: z.number().int().min(1).optional().describe('1-based matching key occurrence to inspect. Defaults to 1.'),
                ...keyFilterSchema,
            }),
        }),
    ]

    if (options.keyOnly) {
        return keyTools.filter(item => item.name !== 'inspect_json_kv_file')
    }

    return [...projectFileTools, ...keyTools]
}

async function readProjectTextFile (root: string, filePath: string): Promise<string> {
    const file = await resolveProjectFile(root, filePath)
    const text = await readFile(file.realFilePath, 'utf8')

    if (text.includes('\0')) {
        throw new Error(`${filePath} appears to be a binary file.`)
    }

    return text
}

async function resolveProjectFile (root: string, requestedPath: string): Promise<{
    resolvedPath: string
    realFilePath: string
    normalizedRelativePath: string
    fileStat: import('node:fs').Stats
}> {
    assertProjectPathAllowed(root, requestedPath)
    return resolveSafeProjectFile(root, requestedPath, {
        realPathMessage: `Path resolves outside the run directory: ${requestedPath}`,
    })
}

async function loadManualTransKeys (root: string, manualTransFile: string): Promise<{
    relativePath: string
    realFilePath: string
    size: number
    mtimeMs: number
    keys: string[]
}> {
    const file = await resolveProjectFile(root, manualTransFile)
    const cached = keyFileCache.get(file.realFilePath)

    if (cached && cached.size === file.fileStat.size && cached.mtimeMs === file.fileStat.mtimeMs) {
        return {
            relativePath: file.normalizedRelativePath,
            realFilePath: cached.realFilePath,
            size: cached.size,
            mtimeMs: cached.mtimeMs,
            keys: cached.keys,
        }
    }

    const text = await readFile(file.realFilePath, 'utf8')

    if (text.includes('\0')) {
        throw new Error(`${file.normalizedRelativePath} appears to be a binary file.`)
    }

    const parsed = parseJsonText(text, file.normalizedRelativePath)

    if (!isPlainObject(parsed)) {
        throw new Error(`${file.normalizedRelativePath} must be a JSON object with top-level keys.`)
    }

    const keys = Object.keys(parsed)
    keyFileCache.set(file.realFilePath, {
        realFilePath: file.realFilePath,
        size: file.fileStat.size,
        mtimeMs: file.fileStat.mtimeMs,
        keys,
    })

    return {
        relativePath: file.normalizedRelativePath,
        realFilePath: file.realFilePath,
        size: file.fileStat.size,
        mtimeMs: file.fileStat.mtimeMs,
        keys,
    }
}

export async function loadManualTransKeyItems (
    root: string,
    manualTransFile: string,
    options: KeyFilterOptions = {},
): Promise<{
    relativePath: string
    originalKeys: string[]
    keyItems: ManualTransKeyItem[]
    filter: KeyFilter
}> {
    const keySet = await loadManualTransKeySet(root, manualTransFile, options)

    return {
        relativePath: keySet.relativePath,
        originalKeys: keySet.originalKeys,
        keyItems: keySet.keyItems,
        filter: keySet.filter,
    }
}

async function loadManualTransKeySet (
    root: string,
    manualTransFile: string,
    options: KeyFilterOptions,
    scope?: KeyInspectionScope,
): Promise<KeySet> {
    const keyFile = await loadManualTransKeys(root, manualTransFile)
    const filter = normalizeKeyFilterOptions(options)
    const scopeOrNull = scope ?? null

    if (!filter.enabled) {
        const keyItems = keyFile.keys.map((key, originalIndex) => ({
            key,
            originalIndex,
            filteredIndex: scopeOrNull?.key_index_to_filtered_index?.[originalIndex] ?? originalIndex,
        }))
        const scopedKeyItems = applyKeyInspectionScope(keyItems, scopeOrNull)

        return {
            ...keyFile,
            originalKeys: keyFile.keys,
            keyItems: scopedKeyItems,
            keys: scopedKeyItems.map(item => item.key),
            filter,
            scope: scopeOrNull,
        }
    }

    const cacheKey = [
        keyFile.realFilePath,
        keyFile.size,
        keyFile.mtimeMs,
        filter.language,
    ].join('\0')
    const cached = keyFilterCache.get(cacheKey)

    if (cached) {
        const scopedKeyItems = applyKeyInspectionScope(cached.keyItems, scopeOrNull)

        return {
            ...keyFile,
            originalKeys: keyFile.keys,
            keyItems: scopedKeyItems,
            keys: scopedKeyItems.map(item => item.key),
            filter,
            scope: scopeOrNull,
        }
    }

    const keyItems = await filterManualTransKeyItems(root, keyFile.keys, filter.language)
    keyFilterCache.set(cacheKey, {
        realFilePath: keyFile.realFilePath,
        size: keyFile.size,
        mtimeMs: keyFile.mtimeMs,
        filterLanguage: filter.language,
        keyItems,
    })

    const scopedKeyItems = applyKeyInspectionScope(keyItems, scopeOrNull)

    return {
        ...keyFile,
        originalKeys: keyFile.keys,
        keyItems: scopedKeyItems,
        keys: scopedKeyItems.map(item => item.key),
        filter,
        scope: scopeOrNull,
    }
}

function formatKeySetHeader (keySet: KeySet): string[] {
    const header = [
        `File: ${keySet.relativePath}`,
        `Total keys: ${keySet.originalKeys.length}`,
        `Filter enabled: ${keySet.filter.enabled}`,
        `Filter language: ${keySet.filter.language}`,
        `Current keys: ${keySet.keys.length}`,
    ]

    if (keySet.scope) {
        header.push(
            `Scope: filtered indexes ${Math.max(0, keySet.scope.batch_start_index - keySet.scope.context_window)}-${keySet.scope.batch_end_index + keySet.scope.context_window}`,
            `Batch: filtered indexes ${keySet.scope.batch_start_index}-${keySet.scope.batch_end_index}`,
        )
    }

    return header
}

function withKeyFilterMetadata (result: object, rootKey: string, keySet: KeySet): object {
    const output = result as Record<string, unknown>
    const section = output[rootKey]

    if (!section || typeof section !== 'object' || Array.isArray(section)) {
        return output
    }

    output[rootKey] = {
        ...(section as Record<string, unknown>),
        original_total_keys: keySet.originalKeys.length,
        filter_enabled: keySet.filter.enabled,
        filter_language: keySet.filter.language,
        current_keys: keySet.keys.length,
        scope: keySet.scope,
    }

    return output
}

function normalizeToolKeyFilterInput (input: KeyFilterOptions, options: CreateInspectionToolsOptions): KeyFilterOptions {
    if (options.allowUnfiltered !== false) {
        return input
    }

    return {
        ...input,
        filter: true,
    }
}

function applyKeyInspectionScope (keyItems: FilteredManualTransKey[], scope: KeyInspectionScope|null): FilteredManualTransKey[] {
    if (!scope) {
        return keyItems
    }

    if (scope.allowed_key_indices?.length) {
        const allowedKeyIndices = new Set(scope.allowed_key_indices)
        return keyItems.filter(item => allowedKeyIndices.has(item.originalIndex))
    }

    const startIndex = Math.max(0, scope.batch_start_index - scope.context_window)
    const endIndex = Math.min(keyItems.length - 1, scope.batch_end_index + scope.context_window)

    if (endIndex < startIndex) {
        return []
    }

    return keyItems.slice(startIndex, endIndex + 1)
}

function selectKeyItemsByFilteredRange (
    keyItems: FilteredManualTransKey[],
    startIndex: number,
    endIndex: number,
): FilteredManualTransKey[] {
    return keyItems.filter(item => item.filteredIndex >= startIndex && item.filteredIndex <= endIndex)
}

function isPlainObject (value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
}

async function inspectJsonKvFile (root: string, manualTransFile: string, sampleLimit: number): Promise<string> {
    const file = await resolveProjectFile(root, manualTransFile)
    const text = await readFile(file.realFilePath, 'utf8')
    const duplicateSummary = findPossibleDuplicateJsonKeys(text, sampleLimit)
    const baseResult = {
        format: 'json_kv_translation',
        file: file.normalizedRelativePath,
        valid_json: false,
        top_level_type: 'unknown',
        entry_count: 0,
        string_key_ratio: 0,
        string_value_ratio: 0,
        key_value_equal_ratio: 0,
        empty_key_count: 0,
        empty_value_count: 0,
        non_string_value_count: 0,
        key_value_different_count: 0,
        possible_duplicate_key_count: duplicateSummary.possibleDuplicateKeyCount,
        warnings: [] as string[],
        samples: {
            empty_keys: [] as string[],
            empty_values: [] as string[],
            non_string_values: [] as Array<{ key: string, value_type: string }>,
            key_value_different: [] as string[],
            possible_duplicate_keys: duplicateSummary.samples,
        },
    }

    let parsed: unknown

    try {
        parsed = JSON.parse(text)
        baseResult.valid_json = true
    } catch (error) {
        baseResult.top_level_type = 'invalid_json'
        baseResult.warnings.push('Invalid JSON.')
        return JSON.stringify({
            ...baseResult,
            parse_error: error instanceof Error ? error.message : String(error),
        }, null, 2)
    }

    baseResult.top_level_type = getJsonTopLevelType(parsed)

    if (!isPlainObject(parsed)) {
        baseResult.warnings.push('Top-level JSON value is not an object.')

        if (baseResult.possible_duplicate_key_count > 0) {
            baseResult.warnings.push('Possible duplicate keys were detected before parsing.')
        }

        return JSON.stringify(baseResult, null, 2)
    }

    const entries = Object.entries(parsed)
    let stringValueCount = 0
    let keyValueEqualCount = 0

    baseResult.entry_count = entries.length
    baseResult.string_key_ratio = entries.length === 0 ? 1 : 1

    for (const [key, value] of entries) {
        if (key === '') {
            baseResult.empty_key_count += 1
            pushSample(baseResult.samples.empty_keys, key, sampleLimit)
        }

        if (typeof value !== 'string') {
            baseResult.non_string_value_count += 1
            pushSample(baseResult.samples.non_string_values, {
                key,
                value_type: getJsonTopLevelType(value),
            }, sampleLimit)
            continue
        }

        stringValueCount += 1

        if (value === '') {
            baseResult.empty_value_count += 1
            pushSample(baseResult.samples.empty_values, key, sampleLimit)
        }

        if (key === value) {
            keyValueEqualCount += 1
        } else {
            baseResult.key_value_different_count += 1
            pushSample(baseResult.samples.key_value_different, key, sampleLimit)
        }
    }

    baseResult.string_value_ratio = ratio(stringValueCount, entries.length)
    baseResult.key_value_equal_ratio = ratio(keyValueEqualCount, entries.length)

    if (baseResult.non_string_value_count > 0) {
        baseResult.warnings.push('Some values are not strings.')
    }

    if (baseResult.empty_key_count > 0) {
        baseResult.warnings.push('Some keys are empty strings.')
    }

    if (baseResult.empty_value_count > 0) {
        baseResult.warnings.push('Some values are empty strings.')
    }

    if (baseResult.key_value_different_count > 0) {
        baseResult.warnings.push('Some values are already different from keys.')
    }

    if (baseResult.possible_duplicate_key_count > 0) {
        baseResult.warnings.push('Possible duplicate keys were detected before parsing.')
    }

    return JSON.stringify(baseResult, null, 2)
}

function loadKuromojiRuntime (): KuromojiRuntime {
    const loaded = kuromoji as Partial<KuromojiRuntime>

    if (typeof loaded.builder !== 'function') {
        throw new Error('Failed to load kuromoji runtime: builder export is unavailable.')
    }

    return loaded as KuromojiRuntime
}

function getKuromojiTokenizer (): Promise<Tokenizer<IpadicFeatures>> {
    if (!kuromojiTokenizerPromise) {
        kuromojiTokenizerPromise = new Promise((resolve, reject) => {
            kuromojiRuntime.builder({ dicPath: KUROMOJI_DIC_PATH }).build((error, tokenizer) => {
                if (error) {
                    reject(error)
                    return
                }

                resolve(tokenizer)
            })
        })
    }

    return kuromojiTokenizerPromise
}

async function analyzeKeyFrequencyProfile (file: string, keys: string[], tokenizer: Tokenizer<IpadicFeatures>): Promise<object> {
    const termCounts = new Map<string, { token: string, count: number, pos: string, kind: string }>()
    const phraseCounts = new Map<string, number>()
    let analyzedKeyCount = 0
    let skippedEmptyKeyCount = 0
    let skippedNumericOnlyKeyCount = 0
    let skippedPathLikeKeyCount = 0
    let skippedControlOrGarbledKeyCount = 0
    const yieldController = createYieldController()

    for (const [index, key] of keys.entries()) {
        await yieldController.maybeYield(index)
        const cleaned = cleanKeyForFrequencyAnalysis(key)

        if (!cleaned.text) {
            skippedEmptyKeyCount += cleaned.reason === 'empty' ? 1 : 0
            skippedNumericOnlyKeyCount += cleaned.reason === 'numeric_only' ? 1 : 0
            skippedPathLikeKeyCount += cleaned.reason === 'path_like' ? 1 : 0
            skippedControlOrGarbledKeyCount += cleaned.reason === 'control_or_garbled' ? 1 : 0
            continue
        }

        const tokens = tokenizer.tokenize(cleaned.text)
            .map(token => normalizeFrequencyToken(token))
            .filter((token): token is FrequencyToken => token !== null)

        if (tokens.length === 0) {
            continue
        }

        analyzedKeyCount += 1

        for (const token of tokens) {
            const current = termCounts.get(token.text)

            if (current) {
                current.count += 1
            } else {
                termCounts.set(token.text, {
                    token: token.text,
                    count: 1,
                    pos: token.pos,
                    kind: token.kind,
                })
            }
        }

        collectRepeatedPhrases(tokens.map(token => token.text), phraseCounts)
    }

    const terms = Array.from(termCounts.values())
    const phrases = Array.from(phraseCounts.entries())
        .filter(([, count]) => count >= MIN_REPEATED_PHRASE_COUNT)
        .map(([phrase, count]) => ({ phrase, count }))
        .sort((left, right) => right.count - left.count || left.phrase.localeCompare(right.phrase))
    const highFrequencyTermLikeCount = terms.filter(term => term.count >= MIN_HIGH_FREQUENCY_TERM_COUNT).length
    const repeatedShortPhraseCount = phrases.length
    const topRepeatedPhraseCount = phrases[0]?.count ?? 0
    const katakanaTokenCount = terms.filter(term => isKatakanaToken(term.token)).length
    const capitalOrLatinTokenCount = terms.filter(term => isCapitalOrLatinToken(term.token)).length
    const warnings = buildKeyFrequencyWarnings({
        analyzedKeyCount,
        totalKeys: keys.length,
        skippedPathLikeKeyCount,
        skippedControlOrGarbledKeyCount,
        repeatedShortPhraseCount,
    })

    return {
        frequency_profile: {
            file,
            total_keys: keys.length,
            analyzed_key_count: analyzedKeyCount,
            cleaned_key_ratio: ratioOrZero(analyzedKeyCount, keys.length),
            unique_token_estimate: termCounts.size,
            high_frequency_term_like_count: highFrequencyTermLikeCount,
            repeated_short_phrase_count: repeatedShortPhraseCount,
            katakana_token_count: katakanaTokenCount,
            capital_or_latin_token_count: capitalOrLatinTokenCount,
            top_repeated_phrase_count: topRepeatedPhraseCount,
            term_density_level: getTermDensityLevel(highFrequencyTermLikeCount, repeatedShortPhraseCount, analyzedKeyCount),
            skipped_empty_key_count: skippedEmptyKeyCount,
            skipped_numeric_only_key_count: skippedNumericOnlyKeyCount,
            skipped_path_like_key_count: skippedPathLikeKeyCount,
            skipped_control_or_garbled_key_count: skippedControlOrGarbledKeyCount,
        },
        top_terms: terms
            .filter(term => term.count >= 2)
            .sort((left, right) => right.count - left.count || left.token.localeCompare(right.token))
            .slice(0, TOP_FREQUENCY_TERMS),
        top_phrases: phrases.slice(0, TOP_FREQUENCY_PHRASES),
        warnings,
    }
}

type FrequencyToken = {
    text: string
    pos: string
    kind: string
}

function cleanKeyForFrequencyAnalysis (key: string): { text: string|null, reason: string|null } {
    const trimmed = key.replace(/\s+/g, ' ').trim()

    if (trimmed.length === 0) {
        return { text: null, reason: 'empty' }
    }

    if (/^[-+]?\d+(?:\.\d+)?$/.test(trimmed)) {
        return { text: null, reason: 'numeric_only' }
    }

    if (isPathLikeKey(trimmed)) {
        return { text: null, reason: 'path_like' }
    }

    if (Array.from(trimmed).some(char => isControlChar(char)) || isGarbledOrBinaryLikeKey(trimmed)) {
        return { text: null, reason: 'control_or_garbled' }
    }

    const text = trimmed
        .replace(/(?:\\[A-Za-z]+\[\d+\]|\\[A-Za-z]|%[sdifjoO]|%\d+|\{[^{}]+\}|<[^<>]+>|\$\w+)/g, ' ')
        .replace(/[「」『』【】［］\[\]（）(){}<>]/g, ' ')
        .replace(/[、。！？!?：:；;,.，・|｜~～=＋+*_#@^`"“”]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()

    if (text.length === 0) {
        return { text: null, reason: 'empty' }
    }

    if (!Array.from(text).some(char => isHiraganaChar(char) || isKatakanaChar(char) || isHanChar(char) || isLatinChar(char))) {
        return { text: null, reason: 'empty' }
    }

    return { text, reason: null }
}

function normalizeFrequencyToken (token: IpadicFeatures): FrequencyToken|null {
    if (!isFrequencyPos(token)) {
        return null
    }

    const rawText = token.basic_form && token.basic_form !== '*' ? token.basic_form : token.surface_form
    const text = normalizeTokenText(rawText)

    if (text.length === 0 || isFrequencyStopword(text) || /^[-+]?\d+(?:\.\d+)?$/.test(text)) {
        return null
    }

    if (text.length === 1 && !isHanChar(text) && !isKatakanaChar(text) && !/[A-Z]/.test(text)) {
        return null
    }

    if (isHiraganaOnlyToken(text) || isLowercaseShortLatinToken(text)) {
        return null
    }

    if (!Array.from(text).some(char => isHiraganaChar(char) || isKatakanaChar(char) || isHanChar(char) || isLatinChar(char))) {
        return null
    }

    return {
        text,
        pos: token.pos,
        kind: getFrequencyTokenKind(text, token),
    }
}

function isFrequencyPos (token: IpadicFeatures): boolean {
    if (token.word_type === 'UNKNOWN') {
        return true
    }

    return token.pos === '名詞' || token.pos === '動詞' || token.pos === '形容詞' || token.pos === '副詞'
}

function normalizeTokenText (text: string): string {
    return text
        .replace(/^[\s"'“”‘’`]+|[\s"'“”‘’`]+$/g, '')
        .replace(/[Ａ-Ｚａ-ｚ０-９]/g, char => String.fromCharCode(char.charCodeAt(0) - 0xFEE0))
        .trim()
}

function getFrequencyTokenKind (text: string, token: IpadicFeatures): string {
    if (isKatakanaToken(text)) {
        return 'katakana'
    }

    if (isCapitalOrLatinToken(text)) {
        return 'latin'
    }

    if (token.word_type === 'UNKNOWN') {
        return 'unknown'
    }

    return 'japanese'
}

function collectRepeatedPhrases (tokens: string[], phraseCounts: Map<string, number>): void {
    for (let size = 2; size <= 4; size += 1) {
        if (tokens.length < size) {
            continue
        }

        for (let index = 0; index <= tokens.length - size; index += 1) {
            const phraseTokens = tokens.slice(index, index + size)

            if (new Set(phraseTokens).size === 1) {
                continue
            }

            const phrase = phraseTokens.join(' ')
            phraseCounts.set(phrase, (phraseCounts.get(phrase) ?? 0) + 1)
        }
    }
}

function getTermDensityLevel (highFrequencyTermLikeCount: number, repeatedShortPhraseCount: number, analyzedKeyCount: number): string {
    const highFrequencyRatio = ratioOrZero(highFrequencyTermLikeCount, analyzedKeyCount)

    if (highFrequencyTermLikeCount >= 150 || repeatedShortPhraseCount >= 80 || highFrequencyRatio >= 0.015) {
        return 'high'
    }

    if (highFrequencyTermLikeCount >= 40 || repeatedShortPhraseCount >= 20 || highFrequencyRatio >= 0.006) {
        return 'medium'
    }

    return 'low'
}

function buildKeyFrequencyWarnings (input: {
    analyzedKeyCount: number
    totalKeys: number
    skippedPathLikeKeyCount: number
    skippedControlOrGarbledKeyCount: number
    repeatedShortPhraseCount: number
}): string[] {
    const warnings: string[] = []

    if (ratioOrZero(input.analyzedKeyCount, input.totalKeys) < 0.25) {
        warnings.push('Few keys remained after cleaning; inspect language mix and protection settings before relying on frequency results.')
    }

    if (ratioOrZero(input.skippedPathLikeKeyCount, input.totalKeys) >= 0.05) {
        warnings.push('Many keys were skipped as paths or asset identifiers; keep resource names protected.')
    }

    if (ratioOrZero(input.skippedControlOrGarbledKeyCount, input.totalKeys) >= 0.01) {
        warnings.push('Some keys were skipped as control-character or garbled text; exclude them from translation batches.')
    }

    if (input.repeatedShortPhraseCount >= 20) {
        warnings.push('Repeated short phrases were detected; seed a glossary before translating to improve consistency.')
    }

    return warnings
}

async function analyzeKeyLanguageMix (file: string, keys: string[]): Promise<object> {
    let containsJapanese = 0
    let containsKana = 0
    let containsEnglish = 0
    let containsCjkUnified = 0
    let containsChineseSignal = 0
    let japaneseDominant = 0
    let englishDominant = 0
    let cjkWithoutKanaDominant = 0
    let mixedLanguage = 0
    let likelyTranslatable = 0
    let numericOnly = 0
    let pathLike = 0
    let placeholderLike = 0
    let protectionLikely = 0
    let controlCharKeys = 0
    let garbledOrBinaryLike = 0
    let emptyKeys = 0

    let totalChars = 0
    let hiraganaChars = 0
    let katakanaChars = 0
    let kanjiOrHanChars = 0
    let latinChars = 0
    let digitChars = 0
    let symbolChars = 0
    let controlChars = 0
    let possibleChineseSignalChars = 0
    const yieldController = createYieldController()

    for (const [index, key] of keys.entries()) {
        await yieldController.maybeYield(index)
        if (key.length === 0) {
            emptyKeys += 1
        }

        let keyHiraganaChars = 0
        let keyKatakanaChars = 0
        let keyHanChars = 0
        let keyLatinChars = 0
        let keyDigitChars = 0
        let keySymbolChars = 0
        let keyControlChars = 0

        for (const char of key) {
            totalChars += 1

            if (isHiraganaChar(char)) {
                hiraganaChars += 1
                keyHiraganaChars += 1
            } else if (isKatakanaChar(char)) {
                katakanaChars += 1
                keyKatakanaChars += 1
            } else if (isHanChar(char)) {
                kanjiOrHanChars += 1
                keyHanChars += 1
            } else if (isLatinChar(char)) {
                latinChars += 1
                keyLatinChars += 1
            } else if (isDigitChar(char)) {
                digitChars += 1
                keyDigitChars += 1
            } else {
                symbolChars += 1
                keySymbolChars += 1

                if (isControlChar(char)) {
                    controlChars += 1
                    keyControlChars += 1
                }
            }
        }

        const keyJapaneseChars = keyHiraganaChars + keyKatakanaChars + keyHanChars
        const keyHasKana = keyHiraganaChars + keyKatakanaChars > 0
        const keyHasJapanese = keyJapaneseChars > 0
        const keyHasEnglish = keyLatinChars > 0
        const keyHasHan = keyHanChars > 0
        const keyHasChineseSignal = keyHasHan && !keyHasKana
        const keyHasControl = keyControlChars > 0
        const keyIsNumericOnly = /^[-+]?\d+(?:\.\d+)?$/.test(key)
        const keyIsPathLike = isPathLikeKey(key)
        const keyIsPlaceholderLike = isPlaceholderLikeKey(key)
        const symbolRatio = key.length === 0 ? 0 : keySymbolChars / Array.from(key).length
        const keyIsGarbledOrBinaryLike = keyHasControl || symbolRatio >= 0.45 && keyLatinChars + keySymbolChars > keyJapaneseChars + keyDigitChars

        if (keyHasJapanese) {
            containsJapanese += 1
        }

        if (keyHasKana) {
            containsKana += 1
        }

        if (keyHasEnglish) {
            containsEnglish += 1
        }

        if (keyHasHan) {
            containsCjkUnified += 1
        }

        if (keyHasChineseSignal) {
            containsChineseSignal += 1
            possibleChineseSignalChars += keyHanChars
        }

        if (keyJapaneseChars > keyLatinChars && keyJapaneseChars > keyDigitChars) {
            japaneseDominant += 1
        } else if (keyLatinChars > keyJapaneseChars && keyLatinChars > keyDigitChars) {
            englishDominant += 1
        }

        if (keyHasChineseSignal && keyHanChars > keyLatinChars && keyHanChars > keyDigitChars) {
            cjkWithoutKanaDominant += 1
        }

        if (keyHasJapanese && keyHasEnglish) {
            mixedLanguage += 1
        }

        if (keyHasJapanese && !keyHasControl && !keyIsPathLike && !keyIsPlaceholderLike) {
            likelyTranslatable += 1
        }

        if (keyIsNumericOnly) {
            numericOnly += 1
        }

        if (keyIsPathLike) {
            pathLike += 1
        }

        if (keyIsPlaceholderLike) {
            placeholderLike += 1
        }

        if (keyIsPathLike || keyIsPlaceholderLike || keyHasControl || symbolRatio >= 0.35) {
            protectionLikely += 1
        }

        if (keyHasControl) {
            controlCharKeys += 1
        }

        if (keyIsGarbledOrBinaryLike) {
            garbledOrBinaryLike += 1
        }
    }

    const japaneseCharCount = hiraganaChars + katakanaChars + kanjiOrHanChars
    const warnings = buildKeyLanguageMixWarnings({
        containsChineseSignal,
        symbolRatio: ratioOrZero(symbolChars, totalChars),
        controlCharKeys,
        pathLike,
        mixedLanguage,
        garbledOrBinaryLike,
        totalKeys: keys.length,
    })

    return {
        key_language_mix: {
            file,
            total_keys: keys.length,
            key_level: {
                contains_japanese_ratio: ratioOrZero(containsJapanese, keys.length),
                contains_kana_ratio: ratioOrZero(containsKana, keys.length),
                contains_english_ratio: ratioOrZero(containsEnglish, keys.length),
                contains_cjk_unified_ratio: ratioOrZero(containsCjkUnified, keys.length),
                contains_chinese_signal_ratio: ratioOrZero(containsChineseSignal, keys.length),
                japanese_dominant_key_ratio: ratioOrZero(japaneseDominant, keys.length),
                english_dominant_key_ratio: ratioOrZero(englishDominant, keys.length),
                cjk_without_kana_dominant_key_ratio: ratioOrZero(cjkWithoutKanaDominant, keys.length),
                mixed_language_key_ratio: ratioOrZero(mixedLanguage, keys.length),
            },
            char_level: {
                japanese_char_ratio: ratioOrZero(japaneseCharCount, totalChars),
                latin_char_ratio: ratioOrZero(latinChars, totalChars),
                possible_chinese_signal_char_ratio: ratioOrZero(possibleChineseSignalChars, totalChars),
                digit_ratio: ratioOrZero(digitChars, totalChars),
                symbol_ratio: ratioOrZero(symbolChars, totalChars),
                control_char_ratio: ratioOrZero(controlChars, totalChars),
            },
            script_breakdown: {
                hiragana_ratio: ratioOrZero(hiraganaChars, totalChars),
                katakana_ratio: ratioOrZero(katakanaChars, totalChars),
                kanji_or_han_ratio: ratioOrZero(kanjiOrHanChars, totalChars),
                latin_ratio: ratioOrZero(latinChars, totalChars),
                digit_ratio: ratioOrZero(digitChars, totalChars),
                symbol_ratio: ratioOrZero(symbolChars, totalChars),
                cjk_unified_ratio: ratioOrZero(kanjiOrHanChars, totalChars),
            },
            translation_hints: {
                likely_translatable_key_ratio: ratioOrZero(likelyTranslatable, keys.length),
                numeric_only_key_ratio: ratioOrZero(numericOnly, keys.length),
                path_like_key_ratio: ratioOrZero(pathLike, keys.length),
                placeholder_like_key_ratio: ratioOrZero(placeholderLike, keys.length),
                protection_likely_key_ratio: ratioOrZero(protectionLikely, keys.length),
                control_char_key_ratio: ratioOrZero(controlCharKeys, keys.length),
                garbled_or_binary_like_key_ratio: ratioOrZero(garbledOrBinaryLike, keys.length),
                empty_key_count: emptyKeys,
            },
            warnings,
        },
    }
}

function buildKeyLanguageMixWarnings (input: {
    containsChineseSignal: number
    symbolRatio: number
    controlCharKeys: number
    pathLike: number
    mixedLanguage: number
    garbledOrBinaryLike: number
    totalKeys: number
}): string[] {
    const warnings: string[] = []

    if (input.containsChineseSignal > 0) {
        warnings.push('Some keys contain CJK text without kana; review for possible Chinese text or all-kanji Japanese segments.')
    }

    if (input.symbolRatio >= 0.15) {
        warnings.push('High symbol ratio; placeholder, control-code, or path protection is likely needed before translation.')
    }

    if (ratioOrZero(input.controlCharKeys, input.totalKeys) >= 0.01) {
        warnings.push('Some keys contain control characters; inspect them before sending text to translation.')
    }

    if (ratioOrZero(input.pathLike, input.totalKeys) >= 0.05) {
        warnings.push('Many keys look like file paths or asset identifiers; avoid translating resource names.')
    }

    if (ratioOrZero(input.mixedLanguage, input.totalKeys) >= 0.1) {
        warnings.push('Many keys mix Japanese/CJK and Latin text; preserve product names, commands, and engine identifiers.')
    }

    if (ratioOrZero(input.garbledOrBinaryLike, input.totalKeys) >= 0.01) {
        warnings.push('Some keys look garbled or binary-like; exclude or inspect them before translation.')
    }

    return warnings
}

function getJsonTopLevelType (value: unknown): string {
    if (value === null) {
        return 'null'
    }

    if (Array.isArray(value)) {
        return 'array'
    }

    return typeof value
}

function ratio (count: number, total: number): number {
    if (total === 0) {
        return 1
    }

    return Number((count / total).toFixed(6))
}

function ratioOrZero (count: number, total: number): number {
    if (total === 0) {
        return 0
    }

    return Number((count / total).toFixed(6))
}

function isHiraganaChar (char: string): boolean {
    return /\p{Script=Hiragana}/u.test(char)
}

function isKatakanaChar (char: string): boolean {
    return /\p{Script=Katakana}/u.test(char)
}

function isHanChar (char: string): boolean {
    return /\p{Script=Han}/u.test(char)
}

function isLatinChar (char: string): boolean {
    return /\p{Script=Latin}/u.test(char)
}

function isDigitChar (char: string): boolean {
    return /\p{Decimal_Number}/u.test(char)
}

function isControlChar (char: string): boolean {
    return /[\u0000-\u001F\u007F-\u009F]/u.test(char)
}

function isPathLikeKey (key: string): boolean {
    return /[\\/]/.test(key) || /\.(?:png|jpe?g|webp|gif|ogg|wav|mp3|json|rxdata|rvdata2?|rgss2?a|txt|csv)$/i.test(key)
}

function isPlaceholderLikeKey (key: string): boolean {
    return /(?:\\[A-Za-z]+\[\d+\]|\\[A-Za-z]|%[sdifjoO]|%\d+|\{[^{}]+\}|<[^<>]+>|\$\w+)/.test(key)
}

function isGarbledOrBinaryLikeKey (key: string): boolean {
    const chars = Array.from(key)

    if (chars.length === 0) {
        return false
    }

    const symbolCount = chars.filter(char => !isHiraganaChar(char) && !isKatakanaChar(char) && !isHanChar(char) && !isLatinChar(char) && !isDigitChar(char)).length
    const languageCount = chars.filter(char => isHiraganaChar(char) || isKatakanaChar(char) || isHanChar(char) || isLatinChar(char)).length

    return symbolCount / chars.length >= 0.45 && symbolCount > languageCount
}

function isKatakanaToken (token: string): boolean {
    return Array.from(token).some(char => isKatakanaChar(char)) && !Array.from(token).some(char => isHiraganaChar(char) || isHanChar(char))
}

function isCapitalOrLatinToken (token: string): boolean {
    return /^[A-Za-z][A-Za-z0-9_-]*$/.test(token) || /[A-Z][A-Za-z0-9_-]*/.test(token)
}

function isHiraganaOnlyToken (token: string): boolean {
    return Array.from(token).length > 0 && Array.from(token).every(char => isHiraganaChar(char))
}

function isLowercaseShortLatinToken (token: string): boolean {
    return /^[a-z]{1,3}$/.test(token)
}

function isFrequencyStopword (token: string): boolean {
    return frequencyStopwords.has(token)
}

const frequencyStopwords = new Set([
    'する',
    'いる',
    'ある',
    'なる',
    'ない',
    'れる',
    'られる',
    'こと',
    'もの',
    'ため',
    'よう',
    'これ',
    'それ',
    'あれ',
    'ここ',
    'そこ',
    'どこ',
])

function pushSample<T> (samples: T[], value: T, limit: number): void {
    if (samples.length < limit) {
        samples.push(value)
    }
}

function findPossibleDuplicateJsonKeys (text: string, sampleLimit: number): {
    possibleDuplicateKeyCount: number
    samples: string[]
} {
    const keys = scanTopLevelJsonObjectKeys(text)
    const seen = new Set<string>()
    const duplicates = new Set<string>()

    for (const key of keys) {
        if (seen.has(key)) {
            duplicates.add(key)
        } else {
            seen.add(key)
        }
    }

    return {
        possibleDuplicateKeyCount: duplicates.size,
        samples: Array.from(duplicates).slice(0, sampleLimit),
    }
}

function scanTopLevelJsonObjectKeys (text: string): string[] {
    const keys: string[] = []
    let index = skipWhitespace(text, 0)

    if (text[index] !== '{') {
        return keys
    }

    index += 1

    while (index < text.length) {
        index = skipWhitespace(text, index)

        if (text[index] === '}') {
            return keys
        }

        if (text[index] === ',') {
            index += 1
            continue
        }

        if (text[index] !== '"') {
            return keys
        }

        const parsedKey = parseJsonStringToken(text, index)

        if (!parsedKey) {
            return keys
        }

        index = skipWhitespace(text, parsedKey.endIndex)

        if (text[index] !== ':') {
            return keys
        }

        keys.push(parsedKey.value)
        index = skipJsonValue(text, index + 1)

        if (index < 0) {
            return keys
        }
    }

    return keys
}

function parseJsonStringToken (text: string, startIndex: number): { value: string, endIndex: number }|null {
    let index = startIndex + 1
    let escaped = false

    while (index < text.length) {
        const char = text[index]

        if (escaped) {
            escaped = false
            index += 1
            continue
        }

        if (char === '\\') {
            escaped = true
            index += 1
            continue
        }

        if (char === '"') {
            const rawString = text.slice(startIndex, index + 1)

            try {
                return {
                    value: JSON.parse(rawString) as string,
                    endIndex: index + 1,
                }
            } catch {
                return null
            }
        }

        index += 1
    }

    return null
}

function skipJsonValue (text: string, startIndex: number): number {
    let index = skipWhitespace(text, startIndex)
    let depth = 0
    let inString = false
    let escaped = false

    while (index < text.length) {
        const char = text[index]

        if (inString) {
            if (escaped) {
                escaped = false
            } else if (char === '\\') {
                escaped = true
            } else if (char === '"') {
                inString = false
            }

            index += 1
            continue
        }

        if (char === '"') {
            inString = true
            index += 1
            continue
        }

        if (char === '{' || char === '[') {
            depth += 1
            index += 1
            continue
        }

        if (char === '}' || char === ']') {
            if (depth === 0) {
                return index
            }

            depth -= 1
            index += 1
            continue
        }

        if (char === ',' && depth === 0) {
            return index
        }

        index += 1
    }

    return index
}

function skipWhitespace (text: string, startIndex: number): number {
    let index = startIndex

    while (index < text.length && /\s/.test(text[index])) {
        index += 1
    }

    return index
}

function assertProjectPathAllowed (root: string, requestedPath: string): void {
    const { normalizedRelativePath } = resolveProjectPath(root, requestedPath, {
        rootPathMessage: `Path is outside the project root: ${requestedPath}`,
        rejectRootPathMessage: 'Expected a file path, received the project root.',
    })

    if (sensitiveRootFiles.has(normalizedRelativePath)) {
        throw new Error(`${normalizedRelativePath} is not available to the agent.`)
    }
}

async function listSearchableFiles (root: string): Promise<string[]> {
    const files: string[] = []
    await collectSearchableFiles(root, root, files)
    return files.slice(0, MAX_SEARCH_FILES)
}

async function collectSearchableFiles (root: string, currentDirectory: string, files: string[]): Promise<void> {
    if (files.length >= MAX_SEARCH_FILES) {
        return
    }

    const entries = await readdir(currentDirectory, { withFileTypes: true })

    for (const entry of entries) {
        if (files.length >= MAX_SEARCH_FILES) {
            return
        }

        const entryPath = path.join(currentDirectory, entry.name)

        if (entry.isDirectory()) {
            if (!skippedDirectories.has(entry.name)) {
                await collectSearchableFiles(root, entryPath, files)
            }
            continue
        }

        if (!entry.isFile() || !isSearchableFile(root, entryPath)) {
            continue
        }

        const fileStat = await stat(entryPath)

        if (fileStat.size > MAX_SEARCH_FILE_BYTES) {
            continue
        }

        files.push(entryPath)
    }
}

function isSearchableFile (root: string, filePath: string): boolean {
    const relativePath = toProjectPath(root, filePath)

    if (sensitiveRootFiles.has(relativePath)) {
        return false
    }

    return !skippedExtensions.has(path.extname(filePath).toLowerCase())
}

async function fileContainsNullByte (filePath: string): Promise<boolean> {
    const file = await open(filePath, 'r')

    try {
        const buffer = Buffer.alloc(BINARY_SAMPLE_BYTES)
        const result = await file.read(buffer, 0, BINARY_SAMPLE_BYTES, 0)
        return buffer.subarray(0, result.bytesRead).includes(0)
    } finally {
        await file.close()
    }
}

async function countTextFileLines (filePath: string, fileSize: number): Promise<number> {
    if (fileSize === 0) {
        return 0
    }

    return new Promise((resolve, reject) => {
        let lineCount = 0
        let lastByte: number|null = null
        const stream = createReadStream(filePath)

        stream.on('data', chunk => {
            const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)

            for (const byte of buffer) {
                if (byte === 10) {
                    lineCount += 1
                }
            }

            lastByte = buffer.at(-1) ?? lastByte
        })

        stream.on('error', reject)
        stream.on('end', () => {
            resolve(lastByte === 10 ? lineCount : lineCount + 1)
        })
    })
}

function suggestNextStep (isBinary: boolean, fileSize: number, lineCount: number|null): string {
    if (isBinary) {
        return 'This appears to be binary; do not use text-reading tools.'
    }

    if (fileSize > MAX_SEARCH_FILE_BYTES) {
        return 'File is large; use read_file_lines with targeted ranges.'
    }

    if (lineCount !== null && lineCount > MAX_READ_LINES * 3) {
        return 'Use sample_file_lines first, then read_file_lines for targeted ranges.'
    }

    return 'Use read_file_lines to inspect the relevant range.'
}

function formatKeyItemList (keyItems: FilteredManualTransKey[], options: CreateInspectionToolsOptions = {}): string {
    if (keyItems.length === 0) {
        return '(none)'
    }

    return keyItems.map(item => `- ${formatKeyItem(item, options)}: ${item.key}`).join('\n')
}

function formatIndexedKeyList (keyItems: FilteredManualTransKey[], options: CreateInspectionToolsOptions = {}): string {
    if (keyItems.length === 0) {
        return '(none)'
    }

    return keyItems.map(item => `${formatKeyItem(item, options)}: ${item.key}`).join('\n')
}

function formatKeyItem (item: FilteredManualTransKey, options: CreateInspectionToolsOptions): string {
    return options.exposeRawKeyIndex === true
        ? `filtered_key_index=${item.filteredIndex}, key_index=${item.originalIndex}`
        : `filtered_key_index=${item.filteredIndex}`
}

function createMatcher (pattern: string, isRegex: boolean, caseSensitive: boolean): { test: (text: string) => boolean } {
    if (!isRegex) {
        const needle = caseSensitive ? pattern : pattern.toLowerCase()

        return {
            test: (text: string): boolean => {
                const haystack = caseSensitive ? text : text.toLowerCase()
                return haystack.includes(needle)
            },
        }
    }

    const flags = caseSensitive ? '' : 'i'
    const regex = new RegExp(pattern, flags)

    return {
        test: (text: string): boolean => {
            regex.lastIndex = 0
            return regex.test(text)
        },
    }
}

function splitLines (text: string): string[] {
    return text.replace(/\r\n/g, '\n').split('\n')
}

function formatLineRange (lines: string[], startLine: number, endLine: number): string {
    const output: string[] = []
    const firstLine = Math.max(1, startLine)
    const lastLine = Math.min(lines.length, endLine)

    for (let lineNumber = firstLine; lineNumber <= lastLine; lineNumber += 1) {
        output.push(`${lineNumber.toString().padStart(4, ' ')} | ${lines[lineNumber - 1]}`)
    }

    return output.join('\n')
}

function toProjectPath (root: string, filePath: string): string {
    return toPosixPath(path.relative(root, filePath))
}

function limitToolOutput (output: string): string {
    return truncate(output, MAX_TOOL_OUTPUT_CHARS)
}
