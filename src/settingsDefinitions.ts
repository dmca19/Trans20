import { configSchema, reasoningEffortValues, type AgentConfig, type AgentConfigField, type ReasoningEffort } from './config.js'
import { getTuiLanguageChoices, isTuiLanguage, type TuiLanguage } from './tuiLanguage.js'
import { getTuiText, type TuiText } from './tuiText.js'
import type { Preferences } from './preferences.js'

export type SettingsFieldKind = 'string'|'number'|'nullable-number'|'boolean'|'enum'|'secret'|'tui-language'

export type SettingsFieldId = AgentConfigField|'tuiLanguage'

export type SettingsFieldDefinition = {
    field: SettingsFieldId
    kind: SettingsFieldKind
}

export type SettingsCategoryId = keyof TuiText['settings']['categories']

export type SettingsCategoryDefinition = {
    id: SettingsCategoryId
    fields: SettingsFieldDefinition[]
}

export type SettingsEditValue = AgentConfig[AgentConfigField]|Preferences['tuiLanguage']

export type SettingsEditInput = {
    field: SettingsFieldId
    input: string
    selectedOptionIndex: number
}

type ParseSettingsEditValueResult = { ok: true, value: SettingsEditValue }|{ ok: false, error: string }

export const settingsCategories = [
    {
        id: 'ui',
        fields: [
            { field: 'tuiLanguage', kind: 'tui-language' },
        ],
    },
    {
        id: 'model-api',
        fields: [
            { field: 'url', kind: 'string' },
            { field: 'key', kind: 'secret' },
            { field: 'model', kind: 'string' },
            { field: 'contextWindow', kind: 'number' },
            { field: 'reasoningEffort', kind: 'enum' },
            { field: 'apiRetryAttempts', kind: 'number' },
            { field: 'agentLifecycleMaxRestarts', kind: 'number' },
        ],
    },
    {
        id: 'files',
        fields: [
            { field: 'manualTransFile', kind: 'string' },
        ],
    },
    {
        id: 'glossary',
        fields: [
            { field: 'glossaryWorkerParallelism', kind: 'number' },
            { field: 'glossaryWorkerBatchSize', kind: 'number' },
            { field: 'glossaryWorkerRecursionLimit', kind: 'number' },
            { field: 'glossaryReviewEveryCompletedBatches', kind: 'number' },
            { field: 'glossaryReviewRecursionLimit', kind: 'number' },
            { field: 'enableGlossaryBatchTools', kind: 'boolean' },
            { field: 'enableGlossaryTargetTermRevisionCheck', kind: 'boolean' },
            { field: 'glossarySubmitSummaryMaxChars', kind: 'nullable-number' },
            { field: 'glossarySubmitDeferredNotesMaxItems', kind: 'nullable-number' },
        ],
    },
    {
        id: 'translation',
        fields: [
            { field: 'translationWorkerParallelism', kind: 'number' },
            { field: 'translationWorkerBatchSize', kind: 'number' },
            { field: 'translationWorkerRecursionLimit', kind: 'number' },
            { field: 'enableTranslationMemorySearch', kind: 'boolean' },
            { field: 'enableCharacterGenderWarnings', kind: 'boolean' },
        ],
    },
    {
        id: 'runtime',
        fields: [
            { field: 'rollbackOnFailure', kind: 'boolean' },
            { field: 'enableRunLogs', kind: 'boolean' },
            { field: 'enableDebugLogs', kind: 'boolean' },
        ],
    },
] satisfies SettingsCategoryDefinition[]

export const settingsFieldDefinitions = new Map<SettingsFieldId, SettingsFieldDefinition>(
    settingsCategories.flatMap(category => category.fields.map(field => [field.field, field] as const)),
)

export const maxSettingsFieldCount = Math.max(...settingsCategories.map(category => category.fields.length))

export function getSettingsEditOptions (kind: SettingsFieldKind): string[] {
    if (kind === 'boolean') {
        return ['true', 'false']
    }

    if (kind === 'tui-language') {
        return getTuiLanguageChoices().map(choice => choice.value)
    }

    if (kind === 'enum') {
        return [...reasoningEffortValues]
    }

    return []
}

export function getSettingsFieldRawValue (
    field: SettingsFieldId,
    config: AgentConfig,
    preferences: Preferences,
): SettingsEditValue {
    return field === 'tuiLanguage' ? preferences.tuiLanguage : config[field]
}

export function formatSettingsRawValue (value: SettingsEditValue): string {
    return value === null ? 'null' : String(value)
}

export function parseSettingsEditValue (
    definition: SettingsFieldDefinition,
    edit: SettingsEditInput,
    currentValue: SettingsEditValue,
    text: TuiText = getTuiText(),
): ParseSettingsEditValueResult {
    if (definition.kind === 'boolean') {
        const value = getSettingsEditOptions(definition.kind)[edit.selectedOptionIndex]

        return value === 'true' || value === 'false'
            ? validateSettingsEditValue(definition, value === 'true', text)
            : { ok: false, error: text.settings.invalidValue(getSettingsFieldLabel(definition, text)) }
    }

    if (definition.kind === 'enum') {
        const value = getSettingsEditOptions(definition.kind)[edit.selectedOptionIndex]

        return reasoningEffortValues.includes(value as ReasoningEffort)
            ? validateSettingsEditValue(definition, value as ReasoningEffort, text)
            : { ok: false, error: text.settings.invalidValue(getSettingsFieldLabel(definition, text)) }
    }

    if (definition.kind === 'tui-language') {
        const value = getSettingsEditOptions(definition.kind)[edit.selectedOptionIndex]

        return value && isTuiLanguage(value)
            ? validateSettingsEditValue(definition, value as TuiLanguage, text)
            : { ok: false, error: text.settings.invalidValue(getSettingsFieldLabel(definition, text)) }
    }

    const rawInput = edit.input.trim()

    if (definition.kind === 'secret' && rawInput.length === 0) {
        return validateSettingsEditValue(definition, currentValue, text)
    }

    if (definition.kind === 'number' || definition.kind === 'nullable-number') {
        if (definition.kind === 'nullable-number' && rawInput.toLowerCase() === 'null') {
            return validateSettingsEditValue(definition, null, text)
        }

        if (rawInput.length === 0) {
            return { ok: false, error: text.settings.integerRequired(getSettingsFieldLabel(definition, text), definition.kind === 'nullable-number') }
        }

        const value = Number(rawInput)

        return Number.isInteger(value)
            ? validateSettingsEditValue(definition, value, text)
            : { ok: false, error: text.settings.integerRequired(getSettingsFieldLabel(definition, text), definition.kind === 'nullable-number') }
    }

    if (rawInput.length === 0) {
        return { ok: false, error: text.settings.nonEmptyRequired(getSettingsFieldLabel(definition, text)) }
    }

    return validateSettingsEditValue(definition, rawInput, text)
}

export function getSettingsFieldLabel (definition: SettingsFieldDefinition, text: TuiText): string {
    return text.settings.fields[definition.field]?.label ?? definition.field
}

export function getSettingsFieldDescription (definition: SettingsFieldDefinition, text: TuiText): string {
    return text.settings.fields[definition.field]?.description ?? ''
}

export function formatSettingsDisplayValue (definition: SettingsFieldDefinition, config: AgentConfig, preferences: Preferences, text: TuiText = getTuiText()): string {
    const value = getSettingsFieldRawValue(definition.field, config, preferences)

    if (definition.kind === 'secret') {
        return maskSecret(String(value ?? ''), text)
    }

    return value === null ? 'null' : String(value)
}

function maskSecret (value: string, text: TuiText = getTuiText()): string {
    if (value.length === 0) {
        return text.settings.emptyValue
    }

    if (value.length <= 8) {
        return '********'
    }

    return `${value.slice(0, 3)}...${value.slice(-4)}`
}

function validateSettingsEditValue (
    definition: SettingsFieldDefinition,
    value: SettingsEditValue,
    text: TuiText,
): ParseSettingsEditValueResult {
    if (definition.field === 'tuiLanguage') {
        return { ok: true, value }
    }

    const schema = configSchema.shape[definition.field]
    const result = schema.safeParse(value)

    return result.success
        ? { ok: true, value: result.data as SettingsEditValue }
        : { ok: false, error: text.settings.invalidValue(getSettingsFieldLabel(definition, text)) }
}
