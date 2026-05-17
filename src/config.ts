import path from 'node:path'

import { z } from 'zod'

import { readOptionalJsonFile, writeJsonFile } from './fileUtils.js'

const DEFAULT_CONTEXT_WINDOW = 300000
const DEFAULT_MANUAL_TRANS_FILE = 'input/ManualTransFile.json'
const DEFAULT_REASONING_EFFORT: ReasoningEffort = 'none'
const DEFAULT_GLOSSARY_WORKER_PARALLELISM = 4
const DEFAULT_GLOSSARY_WORKER_BATCH_SIZE = 50
const MAX_GLOSSARY_WORKER_BATCH_SIZE = 500
const DEFAULT_GLOSSARY_WORKER_RECURSION_LIMIT = 80
const DEFAULT_GLOSSARY_REVIEW_EVERY_COMPLETED_BATCHES = 8
const DEFAULT_GLOSSARY_REVIEW_RECURSION_LIMIT = 120
const DEFAULT_ENABLE_GLOSSARY_BATCH_TOOLS = false
const DEFAULT_ENABLE_GLOSSARY_TARGET_TERM_REVISION_CHECK = true
const DEFAULT_ROLLBACK_ON_FAILURE = true
const DEFAULT_GLOSSARY_SUBMIT_SUMMARY_MAX_CHARS: number|null = null
const DEFAULT_GLOSSARY_SUBMIT_DEFERRED_NOTES_MAX_ITEMS: number|null = null
const DEFAULT_TRANSLATION_WORKER_PARALLELISM = 16
const DEFAULT_TRANSLATION_WORKER_BATCH_SIZE = 50
const MAX_TRANSLATION_WORKER_BATCH_SIZE = 100
const DEFAULT_TRANSLATION_WORKER_RECURSION_LIMIT = 120
const DEFAULT_ENABLE_TRANSLATION_MEMORY_SEARCH = true
const DEFAULT_ENABLE_CHARACTER_GENDER_WARNINGS = false
const DEFAULT_API_RETRY_ATTEMPTS = 3
const DEFAULT_AGENT_LIFECYCLE_MAX_RESTARTS = 5
const DEFAULT_ENABLE_RUN_LOGS = true
const DEFAULT_ENABLE_DEBUG_LOGS = true
const RECOMMENDED_AGENT_CONFIG = {
    glossaryWorkerParallelism: 8,
    enableGlossaryBatchTools: false,
    glossaryReviewEveryCompletedBatches: 8,
    glossaryReviewRecursionLimit: 200,
    apiRetryAttempts: 5,
    agentLifecycleMaxRestarts: 5,
    translationWorkerParallelism: 16,
    translationWorkerRecursionLimit: 200,
    rollbackOnFailure: true,
} satisfies RecommendedAgentConfig
export const reasoningEffortValues = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'] as const

export type ReasoningEffort = typeof reasoningEffortValues[number]

export const configSchema = z.object({
    url: z.string().trim().min(1).optional(),
    key: z.string().trim().min(1).optional(),
    model: z.string().trim().min(1).optional(),
    contextWindow: z.number().int().positive().optional(),
    manualTransFile: z.string().trim().min(1).optional(),
    reasoningEffort: z.enum(reasoningEffortValues).optional(),
    glossaryWorkerParallelism: z.number().int().min(1).max(32).optional(),
    glossaryWorkerBatchSize: z.number().int().min(1).max(MAX_GLOSSARY_WORKER_BATCH_SIZE).optional(),
    glossaryWorkerRecursionLimit: z.number().int().positive().optional(),
    glossaryReviewEveryCompletedBatches: z.number().int().min(1).optional(),
    glossaryReviewRecursionLimit: z.number().int().positive().optional(),
    enableGlossaryBatchTools: z.boolean().optional(),
    enableGlossaryTargetTermRevisionCheck: z.boolean().optional(),
    rollbackOnFailure: z.boolean().optional(),
    glossarySubmitSummaryMaxChars: z.number().int().positive().nullable().optional(),
    glossarySubmitDeferredNotesMaxItems: z.number().int().positive().nullable().optional(),
    translationWorkerParallelism: z.number().int().min(1).max(32).optional(),
    translationWorkerBatchSize: z.number().int().min(1).max(MAX_TRANSLATION_WORKER_BATCH_SIZE).optional(),
    translationWorkerRecursionLimit: z.number().int().positive().optional(),
    enableTranslationMemorySearch: z.boolean().optional(),
    enableCharacterGenderWarnings: z.boolean().optional(),
    apiRetryAttempts: z.number().int().min(1).max(10).optional(),
    agentLifecycleMaxRestarts: z.number().int().min(0).max(10).optional(),
    enableRunLogs: z.boolean().optional(),
    enableDebugLogs: z.boolean().optional(),
})

export const taskConfigSnapshotSchema = configSchema
    .omit({
        url: true,
        key: true,
        enableRunLogs: true,
        enableDebugLogs: true,
    })
    .required()

export const partialTaskConfigSnapshotSchema = taskConfigSnapshotSchema.partial()

export type AgentConfig = {
    url: string
    key: string
    model: string
    contextWindow: number
    manualTransFile: string
    reasoningEffort: ReasoningEffort
    glossaryWorkerParallelism: number
    glossaryWorkerBatchSize: number
    glossaryWorkerRecursionLimit: number
    glossaryReviewEveryCompletedBatches: number
    glossaryReviewRecursionLimit: number
    enableGlossaryBatchTools: boolean
    enableGlossaryTargetTermRevisionCheck: boolean
    rollbackOnFailure: boolean
    glossarySubmitSummaryMaxChars: number|null
    glossarySubmitDeferredNotesMaxItems: number|null
    translationWorkerParallelism: number
    translationWorkerBatchSize: number
    translationWorkerRecursionLimit: number
    enableTranslationMemorySearch: boolean
    enableCharacterGenderWarnings: boolean
    apiRetryAttempts: number
    agentLifecycleMaxRestarts: number
    enableRunLogs: boolean
    enableDebugLogs: boolean
}

export type AgentConfigField = keyof AgentConfig
export type RecommendedAgentConfig = Partial<Omit<AgentConfig, RecommendedAgentConfigIgnoredField>>

export type EditableAgentConfigResult = {
    config: AgentConfig
    rawConfig: Partial<AgentConfig>
}

export function createDefaultAgentConfig (): AgentConfig {
    return {
        url: '',
        key: '',
        model: '',
        contextWindow: DEFAULT_CONTEXT_WINDOW,
        manualTransFile: DEFAULT_MANUAL_TRANS_FILE,
        reasoningEffort: DEFAULT_REASONING_EFFORT,
        glossaryWorkerParallelism: DEFAULT_GLOSSARY_WORKER_PARALLELISM,
        glossaryWorkerBatchSize: DEFAULT_GLOSSARY_WORKER_BATCH_SIZE,
        glossaryWorkerRecursionLimit: DEFAULT_GLOSSARY_WORKER_RECURSION_LIMIT,
        glossaryReviewEveryCompletedBatches: DEFAULT_GLOSSARY_REVIEW_EVERY_COMPLETED_BATCHES,
        glossaryReviewRecursionLimit: DEFAULT_GLOSSARY_REVIEW_RECURSION_LIMIT,
        enableGlossaryBatchTools: DEFAULT_ENABLE_GLOSSARY_BATCH_TOOLS,
        enableGlossaryTargetTermRevisionCheck: DEFAULT_ENABLE_GLOSSARY_TARGET_TERM_REVISION_CHECK,
        rollbackOnFailure: DEFAULT_ROLLBACK_ON_FAILURE,
        glossarySubmitSummaryMaxChars: DEFAULT_GLOSSARY_SUBMIT_SUMMARY_MAX_CHARS,
        glossarySubmitDeferredNotesMaxItems: DEFAULT_GLOSSARY_SUBMIT_DEFERRED_NOTES_MAX_ITEMS,
        translationWorkerParallelism: DEFAULT_TRANSLATION_WORKER_PARALLELISM,
        translationWorkerBatchSize: DEFAULT_TRANSLATION_WORKER_BATCH_SIZE,
        translationWorkerRecursionLimit: DEFAULT_TRANSLATION_WORKER_RECURSION_LIMIT,
        enableTranslationMemorySearch: DEFAULT_ENABLE_TRANSLATION_MEMORY_SEARCH,
        enableCharacterGenderWarnings: DEFAULT_ENABLE_CHARACTER_GENDER_WARNINGS,
        apiRetryAttempts: DEFAULT_API_RETRY_ATTEMPTS,
        agentLifecycleMaxRestarts: DEFAULT_AGENT_LIFECYCLE_MAX_RESTARTS,
        enableRunLogs: DEFAULT_ENABLE_RUN_LOGS,
        enableDebugLogs: DEFAULT_ENABLE_DEBUG_LOGS,
    }
}

export async function loadAgentConfig (): Promise<AgentConfig> {
    const configPath = path.join(process.cwd(), 'config.json')
    const fileConfig = await readConfigFile(configPath)
    const config = mergeAgentConfig(fileConfig)

    const missing = [
        ['url', config.url],
        ['key', config.key],
        ['model', config.model],
    ].filter(([, value]) => !value).map(([name]) => name)

    if (missing.length > 0) {
        throw new Error(
            `Missing model config: ${missing.join(', ')}. Create config.json from config.example.json or set OPENAI_BASE_URL, OPENAI_API_KEY, and OPENAI_MODEL.`,
        )
    }

    return config
}

export async function loadEditableAgentConfig (): Promise<EditableAgentConfigResult> {
    const configPath = path.join(process.cwd(), 'config.json')
    const rawConfig = await readConfigFile(configPath)

    return {
        config: mergeAgentConfig(rawConfig),
        rawConfig,
    }
}

export async function saveAgentConfig (config: AgentConfig): Promise<void> {
    const configPath = path.join(process.cwd(), 'config.json')
    const validatedConfig = configSchema.required().parse(config)
    await writeJsonFile(configPath, validatedConfig)
}

export function getRecommendedAgentConfig (): RecommendedAgentConfig {
    return { ...RECOMMENDED_AGENT_CONFIG }
}

export function applyRecommendedAgentConfig (
    config: AgentConfig,
    recommendedConfig: Partial<AgentConfig>,
): AgentConfig {
    const applicableConfig = filterRecommendedAgentConfig(recommendedConfig)

    return configSchema.required().parse({
        ...config,
        ...applicableConfig,
        url: config.url,
        key: config.key,
        model: config.model,
    })
}

function mergeAgentConfig (fileConfig: Partial<AgentConfig>): AgentConfig {
    const defaults = createDefaultAgentConfig()

    return {
        url: fileConfig.url ?? process.env.OPENAI_BASE_URL ?? defaults.url,
        key: fileConfig.key ?? process.env.OPENAI_API_KEY ?? defaults.key,
        model: fileConfig.model ?? process.env.OPENAI_MODEL ?? defaults.model,
        contextWindow: fileConfig.contextWindow ?? defaults.contextWindow,
        manualTransFile: fileConfig.manualTransFile ?? defaults.manualTransFile,
        reasoningEffort: fileConfig.reasoningEffort ?? defaults.reasoningEffort,
        glossaryWorkerParallelism: fileConfig.glossaryWorkerParallelism ?? defaults.glossaryWorkerParallelism,
        glossaryWorkerBatchSize: fileConfig.glossaryWorkerBatchSize ?? defaults.glossaryWorkerBatchSize,
        glossaryWorkerRecursionLimit: fileConfig.glossaryWorkerRecursionLimit ?? defaults.glossaryWorkerRecursionLimit,
        glossaryReviewEveryCompletedBatches: fileConfig.glossaryReviewEveryCompletedBatches ?? defaults.glossaryReviewEveryCompletedBatches,
        glossaryReviewRecursionLimit: fileConfig.glossaryReviewRecursionLimit ?? defaults.glossaryReviewRecursionLimit,
        enableGlossaryBatchTools: fileConfig.enableGlossaryBatchTools ?? defaults.enableGlossaryBatchTools,
        enableGlossaryTargetTermRevisionCheck: fileConfig.enableGlossaryTargetTermRevisionCheck ?? defaults.enableGlossaryTargetTermRevisionCheck,
        rollbackOnFailure: fileConfig.rollbackOnFailure ?? defaults.rollbackOnFailure,
        glossarySubmitSummaryMaxChars: fileConfig.glossarySubmitSummaryMaxChars ?? defaults.glossarySubmitSummaryMaxChars,
        glossarySubmitDeferredNotesMaxItems: fileConfig.glossarySubmitDeferredNotesMaxItems ?? defaults.glossarySubmitDeferredNotesMaxItems,
        translationWorkerParallelism: fileConfig.translationWorkerParallelism ?? defaults.translationWorkerParallelism,
        translationWorkerBatchSize: fileConfig.translationWorkerBatchSize ?? defaults.translationWorkerBatchSize,
        translationWorkerRecursionLimit: fileConfig.translationWorkerRecursionLimit ?? defaults.translationWorkerRecursionLimit,
        enableTranslationMemorySearch: fileConfig.enableTranslationMemorySearch ?? defaults.enableTranslationMemorySearch,
        enableCharacterGenderWarnings: fileConfig.enableCharacterGenderWarnings ?? defaults.enableCharacterGenderWarnings,
        apiRetryAttempts: fileConfig.apiRetryAttempts ?? defaults.apiRetryAttempts,
        agentLifecycleMaxRestarts: fileConfig.agentLifecycleMaxRestarts ?? defaults.agentLifecycleMaxRestarts,
        enableRunLogs: fileConfig.enableRunLogs ?? defaults.enableRunLogs,
        enableDebugLogs: fileConfig.enableDebugLogs ?? defaults.enableDebugLogs,
    }
}

async function readConfigFile (configPath: string): Promise<Partial<AgentConfig>> {
    return readOptionalJsonFile(configPath, configSchema, {})
}

type RecommendedAgentConfigIgnoredField = 'url'|'key'|'model'|'reasoningEffort'

function filterRecommendedAgentConfig (config: Partial<AgentConfig>): RecommendedAgentConfig {
    return Object.fromEntries(
        Object.entries(config).filter(([field, value]) => !isRecommendedAgentConfigIgnoredField(field) && value !== undefined),
    ) as RecommendedAgentConfig
}

function isRecommendedAgentConfigIgnoredField (field: string): field is RecommendedAgentConfigIgnoredField {
    return field === 'url' || field === 'key' || field === 'model' || field === 'reasoningEffort'
}
