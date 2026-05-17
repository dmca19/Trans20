import path from 'node:path'

import { isNodeError, readJsonFile } from './fileUtils.js'
import type { DefaultSourceLanguage, DefaultTargetLanguage } from './defaultTranslationLanguages.js'
import { createPersistentStateStore } from './persistentStateStore.js'

export const GLOSSARY_FILE_PATH = 'glossary.json'

export const termTypeValues = [
    'character',
    'place',
    'item',
    'skill',
    'system_term',
    'faction',
    'title',
    'repeated_phrase',
] as const

export const entryTypeValues = [
    'translation_rule',
    'fact',
    'style',
    'continuity',
] as const

export const translationRuleBasisValues = [
    'semantic',
    'transliteration',
    'project_convention',
    'observed_translation',
] as const

export const termStatusValues = [
    'active',
    'merged',
    'deprecated',
    'rejected',
] as const

export const rejectedReasonValues = [
    'empty_insufficient_evidence',
    'invalid_or_noise',
    'duplicate_or_superseded',
    'low_translation_value',
] as const

export const entryStatusValues = [
    'candidate',
    'approved',
    'rejected',
] as const

export const policyStrengthValues = [
    'required',
    'recommended',
    'informational',
] as const

export const appliesToValues = [
    'term',
    'speaker',
    'mentioned',
    'global',
] as const

export const genderPresentationValueValues = [
    'male',
    'female',
    'other',
    'unknown',
] as const

export const genderPresentationConfidenceValues = [
    'high',
    'medium',
    'low',
] as const

export const mergeProposalStatusValues = [
    'candidate',
    'approved',
    'rejected',
    'applied',
] as const

export const reviewEntryBatchOperationValues = [
    'approve',
    'reject',
    'revise',
    'merge_into',
    'move_to_term',
    'move_to_term_and_approve',
] as const

export type TermType = typeof termTypeValues[number]
export type EntryType = typeof entryTypeValues[number]
export type TranslationRuleBasis = typeof translationRuleBasisValues[number]
export type TermStatus = typeof termStatusValues[number]
export type RejectedReason = typeof rejectedReasonValues[number]
export type EntryStatus = typeof entryStatusValues[number]
export type PolicyStrength = typeof policyStrengthValues[number]
export type AppliesTo = typeof appliesToValues[number]
export type GenderPresentationValue = typeof genderPresentationValueValues[number]
export type GenderPresentationConfidence = typeof genderPresentationConfidenceValues[number]
export type MergeProposalStatus = typeof mergeProposalStatusValues[number]
export type ReviewEntryBatchOperation = typeof reviewEntryBatchOperationValues[number]

export type GenderPresentation = {
    value: GenderPresentationValue
    confidence: GenderPresentationConfidence
    entry_id: string
}

export type GlossaryAliasMap = Record<string, string>

export type SourceSelector = {
    variant_id: 'term'|string
    text: string
}

export type GlossaryTerm = {
    term_id: string
    source_text: string
    source_language: string
    term_type: TermType
    aliases: GlossaryAliasMap
    alias_order: string[]
    next_alias_seq: number
    status: TermStatus
    rejected_reason?: RejectedReason
    merged_into: string|null
    entry_ids: string[]
    gender_presentations?: GenderPresentation[]
    created_at: string
    updated_at: string
    revision: number
    created_by?: string
    created_from?: Record<string, unknown>
    updated_by?: string
    updated_from?: Record<string, unknown>
}

export type GlossaryEntryContent = {
    summary: string
    description: string
}

export type GlossaryEntryTarget = {
    target_language: string
    preferred_translation?: string
    alternative_translations?: string[]
    forbidden_translations?: string[]
}

export type GlossaryEntryApplicability = {
    applies_to: AppliesTo
    domain?: string
    applies_when?: string[]
    does_not_apply_when?: string[]
    source_selectors?: SourceSelector[]
}

export type GlossaryEntryPolicy = {
    strength: PolicyStrength
    requires_context_check?: boolean
    notes?: string[]
}

type GlossaryEntryBase = {
    entry_id: string
    term_id: string
    content: GlossaryEntryContent
    applicability: GlossaryEntryApplicability
    policy: GlossaryEntryPolicy
    status: EntryStatus
    evidence_ids: string[]
    created_by?: string
    created_from?: Record<string, unknown>
    revision: number
    created_at: string
    updated_at: string
}

export type TranslationRuleGlossaryEntry = GlossaryEntryBase & {
    entry_type: 'translation_rule'
    basis: TranslationRuleBasis
    target: GlossaryEntryTarget
}

export type NonTranslationGlossaryEntry = GlossaryEntryBase & {
    entry_type: Exclude<EntryType, 'translation_rule'>
    basis?: never
    target?: never
}

export type GlossaryEntry = TranslationRuleGlossaryEntry|NonTranslationGlossaryEntry

export type GlossaryEvidence = {
    evidence_id: string
    term_id: string
    entry_id: string
    source_ref: {
        source_file_id?: string
        file_id?: string
        key_index?: number
        key_hash?: string
        span?: {
            start: number
            end: number
        }
    }
    quote: string
    context?: {
        before?: string|null
        after?: string|null
    }
    reason: string
    created_by?: string
    created_at: string
}

export type GlossaryMergeProposal = {
    proposal_id: string
    source_term_id: string
    target_term_id: string
    status: MergeProposalStatus
    reason: string
    evidence: {
        source_ref: {
            source_file_id: string
            key_index: number
            key_hash?: string
            span?: {
                start: number
                end: number
            }
        }
        quote: string
        context?: {
            before?: string|null
            after?: string|null
        }
        reason: string
    }[]
    existing_evidence_ids: string[]
    created_by: string
    created_from: Record<string, unknown>
    revision: number
    created_at: string
    updated_at: string
}

export type GlossaryPlanInput = {
    term_extraction_policy: {
        term_types: TermType[]
        entry_types: EntryType[]
        default_term_status: 'active'
        default_entry_status: 'candidate'
        require_evidence: true
        one_entry_one_claim: true
    }
    shared_prompt_context: {
        task: string
        source_language: DefaultSourceLanguage
        target_language: DefaultTargetLanguage
        project_context: string
        domain_hint: string
        focus: string[]
        cautions: string[]
        notes: string[]
    }
}

export type GlossaryPlan = GlossaryPlanInput & {
    plan_id: string
    created_at: string
}

export type ReviewEntryOperation = ReviewEntryBatchOperation|'append'
export type ReviewTermOperation = 'keep_active'|'reject'|'deprecate'|'change_term_type'|'add_aliases'|'remove_invalid_aliases'|'merge_term'|'move_entries'|'set_gender_presentations'
export type ReviewWindowStatus = 'running'|'completed'|'failed'
export type CachedWorkerWriteStatus = 'pending'|'applied'|'conflict'

export type ReviewRevisedEntry = {
    entry_type?: EntryType
    basis?: TranslationRuleBasis
    content?: GlossaryEntryContent
    target?: GlossaryEntryTarget
    policy?: GlossaryEntryPolicy
    applicability?: GlossaryEntryApplicability & {
        source_variant_indexes?: number[]
    }
}

export type ReviewEntryPayload = {
    entry_type: EntryType
    basis?: TranslationRuleBasis
    content: GlossaryEntryContent
    target?: GlossaryEntryTarget
    applicability: GlossaryEntryApplicability
    policy: GlossaryEntryPolicy
}

export type ReviewEntryAction = {
    action_id: string
    operation: ReviewEntryOperation
    entry_id?: string
    term_id?: string
    expected_entry_revision?: number
    target_entry_id?: string
    target_term_id?: string
    entry?: ReviewEntryPayload
    evidence_ids?: string[]
    evidence?: Pick<GlossaryEvidence, 'source_ref'|'quote'|'context'|'reason'>[]
    revised_entry?: ReviewRevisedEntry
    reason: string
}

export type ReviewTermAction = {
    action_id: string
    operation: ReviewTermOperation
    term_id: string
    expected_term_revision?: number
    target_term_id?: string
    term_type?: TermType
    aliases_to_add?: string[]
    aliases_to_remove?: string[]
    entry_ids?: string[]
    gender_presentations?: GenderPresentation[]
    rejected_reason?: RejectedReason
    reason: string
}

export type CachedWorkerWrite = {
    cache_id: string
    tool_name: string
    input: Record<string, unknown>
    status: CachedWorkerWriteStatus
    conflict_reason?: string
    created_at: string
    applied_at?: string
}

export type ReviewConflict = {
    conflict_id: string
    cache_id?: string
    tool_name: string
    reason: string
    input?: Record<string, unknown>
    created_at: string
}

export type GlossaryReviewWindow = {
    review_window_id: string
    status: ReviewWindowStatus
    started_at: string
    completed_at?: string
    failed_at?: string
    last_reviewed_batch_number: number
    completed_batch_numbers: number[]
    pending_entry_actions: ReviewEntryAction[]
    pending_term_actions: ReviewTermAction[]
    summary?: string
    error?: string
}

export type GlossaryReviewLog = {
    review_window_id: string
    status: ReviewWindowStatus
    started_at: string
    completed_at?: string
    summary?: string
    error?: string
    entry_action_count: number
    term_action_count: number
    cache_applied_count: number
    cache_conflict_count: number
}

export type GlossaryReviewState = {
    frozen: boolean
    last_reviewed_batch_number: number
    completed_batches: number[]
    active_window: GlossaryReviewWindow|null
    windows: GlossaryReviewWindow[]
    cached_worker_writes: CachedWorkerWrite[]
    conflicts: ReviewConflict[]
    logs: GlossaryReviewLog[]
}

export type GlossaryState = {
    version: 1
    active_plan_id: string|null
    plans: GlossaryPlan[]
    terms: GlossaryTerm[]
    entries: GlossaryEntry[]
    evidence: GlossaryEvidence[]
    merge_proposals: GlossaryMergeProposal[]
    review: GlossaryReviewState
    meta: {
        created_at: string
        updated_at: string
    }
}

const glossaryStateStore = createPersistentStateStore<GlossaryState>({
    resolveFilePath: resolveGlossaryFilePath,
    loadFromDisk: readGlossaryStateFromDisk,
})

export async function loadGlossaryState (root: string): Promise<GlossaryState> {
    return readGlossaryState(root, state => structuredClone(state) as GlossaryState)
}

export async function readGlossaryState<T> (
    root: string,
    reader: (state: GlossaryState) => Promise<T>|T,
): Promise<T> {
    return glossaryStateStore.read(root, reader)
}

export async function saveGlossaryState (root: string, state: GlossaryState): Promise<void> {
    await glossaryStateStore.save(root, state)
}

export async function updateGlossaryState<T> (
    root: string,
    updater: (state: GlossaryState) => Promise<T>|T,
): Promise<T> {
    return glossaryStateStore.update(root, updater)
}

export async function appendGlossaryPlan (root: string, input: GlossaryPlanInput): Promise<GlossaryPlan> {
    return updateGlossaryState(root, state => {
        const now = new Date().toISOString()
        const plan: GlossaryPlan = {
            ...input,
            plan_id: createNextPlanId(state.plans),
            created_at: now,
        }

        state.plans.push(plan)
        state.active_plan_id = plan.plan_id
        state.meta.updated_at = now

        return plan
    })
}

export async function flushGlossaryState (root: string): Promise<void> {
    await glossaryStateStore.flush(root)
}

export async function flushAllGlossaryStores (): Promise<void> {
    await glossaryStateStore.flushAll()
}

export async function closeGlossaryStore (root: string): Promise<void> {
    await glossaryStateStore.close(root)
}

function createEmptyGlossaryState (): GlossaryState {
    const now = new Date().toISOString()

    return {
        version: 1,
        active_plan_id: null,
        plans: [],
        terms: [],
        entries: [],
        evidence: [],
        merge_proposals: [],
        review: createEmptyReviewState(),
        meta: {
            created_at: now,
            updated_at: now,
        },
    }
}

function createEmptyReviewState (): GlossaryReviewState {
    return {
        frozen: false,
        last_reviewed_batch_number: 0,
        completed_batches: [],
        active_window: null,
        windows: [],
        cached_worker_writes: [],
        conflicts: [],
        logs: [],
    }
}

function resolveGlossaryFilePath (root: string): string {
    return path.join(root, GLOSSARY_FILE_PATH)
}

async function readGlossaryStateFromDisk (filePath: string): Promise<GlossaryState> {
    try {
        const parsedState = await readJsonFile(filePath, GLOSSARY_FILE_PATH)
        return parseGlossaryState(parsedState)
    } catch (error) {
        if (isNodeError(error) && error.code === 'ENOENT') {
            return createEmptyGlossaryState()
        }

        throw error
    }
}

function parseGlossaryState (value: unknown): GlossaryState {
    if (!isRecord(value)) {
        throw new Error(`${GLOSSARY_FILE_PATH} must contain a JSON object.`)
    }

    const now = new Date().toISOString()

    return {
        version: 1,
        active_plan_id: typeof value.active_plan_id === 'string' ? value.active_plan_id : null,
        plans: Array.isArray(value.plans) ? value.plans.filter(isGlossaryPlan) : [],
        terms: readGlossaryArray(value.terms, 'terms', isGlossaryTerm),
        entries: readGlossaryArray(value.entries, 'entries', isGlossaryEntry),
        evidence: Array.isArray(value.evidence) ? value.evidence.filter(isGlossaryEvidence) : [],
        merge_proposals: Array.isArray(value.merge_proposals) ? value.merge_proposals.filter(isGlossaryMergeProposal) : [],
        review: parseReviewState(value.review),
        meta: {
            created_at: readMetaTimestamp(value, 'created_at') ?? now,
            updated_at: readMetaTimestamp(value, 'updated_at') ?? now,
        },
    }
}

function parseReviewState (value: unknown): GlossaryReviewState {
    const fallback = createEmptyReviewState()

    if (!isRecord(value)) {
        return fallback
    }

    return {
        frozen: false,
        last_reviewed_batch_number: readNonNegativeNumber(value.last_reviewed_batch_number, fallback.last_reviewed_batch_number),
        completed_batches: Array.isArray(value.completed_batches)
            ? value.completed_batches.filter((item): item is number => Number.isInteger(item) && item >= 0)
            : fallback.completed_batches,
        active_window: isGlossaryReviewWindow(value.active_window) ? value.active_window : null,
        windows: Array.isArray(value.windows) ? value.windows.filter(isGlossaryReviewWindow) : [],
        cached_worker_writes: [],
        conflicts: Array.isArray(value.conflicts) ? value.conflicts.filter(isReviewConflict) : [],
        logs: Array.isArray(value.logs) ? value.logs.filter(isGlossaryReviewLog) : [],
    }
}

function createNextPlanId (plans: GlossaryPlan[]): string {
    const nextNumber = plans.reduce((max, plan) => {
        const match = /^plan_(\d+)$/.exec(plan.plan_id)

        if (!match) {
            return max
        }

        return Math.max(max, Number(match[1]))
    }, 0) + 1

    return `plan_${nextNumber.toString().padStart(6, '0')}`
}

function isGlossaryPlan (value: unknown): value is GlossaryPlan {
    return isRecord(value)
        && typeof value.plan_id === 'string'
        && typeof value.created_at === 'string'
        && isRecord(value.term_extraction_policy)
        && isRecord(value.shared_prompt_context)
}

function isGlossaryTerm (value: unknown): value is GlossaryTerm {
    return isRecord(value)
        && typeof value.term_id === 'string'
        && typeof value.source_text === 'string'
        && typeof value.source_language === 'string'
        && (!Object.prototype.hasOwnProperty.call(value, 'rejected_reason') || rejectedReasonValues.includes(value.rejected_reason as RejectedReason))
        && isGlossaryAliasMap(value.aliases)
        && Array.isArray(value.alias_order)
        && value.alias_order.every(item => typeof item === 'string')
        && value.alias_order.every(aliasId => isGlossaryAliasMap(value.aliases) && typeof value.aliases[aliasId] === 'string')
        && typeof value.next_alias_seq === 'number'
        && Number.isInteger(value.next_alias_seq)
        && value.next_alias_seq >= 1
}

function isGlossaryEntry (value: unknown): value is GlossaryEntry {
    return isRecord(value)
        && typeof value.entry_id === 'string'
        && typeof value.term_id === 'string'
        && entryTypeValues.includes(value.entry_type as EntryType)
        && isRecord(value.content)
        && typeof value.content.summary === 'string'
        && typeof value.content.description === 'string'
        && isRecord(value.applicability)
        && appliesToValues.includes(value.applicability.applies_to as AppliesTo)
        && isRecord(value.policy)
        && policyStrengthValues.includes(value.policy.strength as PolicyStrength)
        && entryStatusValues.includes(value.status as EntryStatus)
        && isEntrySourceSelectorShapeValid(value)
}

function isGlossaryEvidence (value: unknown): value is GlossaryEvidence {
    return isRecord(value)
        && typeof value.evidence_id === 'string'
        && typeof value.term_id === 'string'
        && typeof value.entry_id === 'string'
}

function isGlossaryMergeProposal (value: unknown): value is GlossaryMergeProposal {
    return isRecord(value)
        && typeof value.proposal_id === 'string'
        && typeof value.source_term_id === 'string'
        && typeof value.target_term_id === 'string'
        && mergeProposalStatusValues.includes(value.status as MergeProposalStatus)
}

function isGlossaryReviewWindow (value: unknown): value is GlossaryReviewWindow {
    return isRecord(value)
        && typeof value.review_window_id === 'string'
        && (value.status === 'running' || value.status === 'completed' || value.status === 'failed')
        && typeof value.started_at === 'string'
        && typeof value.last_reviewed_batch_number === 'number'
        && Array.isArray(value.completed_batch_numbers)
        && Array.isArray(value.pending_entry_actions)
        && Array.isArray(value.pending_term_actions)
}

function isCachedWorkerWrite (value: unknown): value is CachedWorkerWrite {
    return isRecord(value)
        && typeof value.cache_id === 'string'
        && typeof value.tool_name === 'string'
        && isRecord(value.input)
        && (value.status === 'pending' || value.status === 'applied' || value.status === 'conflict')
        && typeof value.created_at === 'string'
}

function isReviewConflict (value: unknown): value is ReviewConflict {
    return isRecord(value)
        && typeof value.conflict_id === 'string'
        && typeof value.tool_name === 'string'
        && typeof value.reason === 'string'
        && typeof value.created_at === 'string'
}

function isGlossaryReviewLog (value: unknown): value is GlossaryReviewLog {
    return isRecord(value)
        && typeof value.review_window_id === 'string'
        && (value.status === 'running' || value.status === 'completed' || value.status === 'failed')
        && typeof value.started_at === 'string'
        && typeof value.entry_action_count === 'number'
        && typeof value.term_action_count === 'number'
        && typeof value.cache_applied_count === 'number'
        && typeof value.cache_conflict_count === 'number'
}

function readGlossaryArray<T> (
    value: unknown,
    field: string,
    guard: (item: unknown) => item is T,
): T[] {
    if (!Array.isArray(value)) {
        return []
    }

    const invalidIndex = value.findIndex(item => !guard(item))

    if (invalidIndex >= 0) {
        throw new Error(`${GLOSSARY_FILE_PATH} contains invalid ${field}[${invalidIndex}]. This project expects the current glossary schema.`)
    }

    return value
}

function isGlossaryAliasMap (value: unknown): value is GlossaryAliasMap {
    return isRecord(value) && Object.values(value).every(alias => typeof alias === 'string')
}

function isEntrySourceSelectorShapeValid (value: Record<string, unknown>): boolean {
    const applicability = value.applicability

    if (!isRecord(applicability)) {
        return false
    }

    if ('source_variant_indexes' in applicability || 'source_variant_texts' in applicability) {
        return false
    }

    if (value.entry_type !== 'translation_rule') {
        return !('source_selectors' in applicability)
    }

    return Array.isArray(applicability.source_selectors)
        && applicability.source_selectors.length > 0
        && applicability.source_selectors.every(selector => (
            isRecord(selector)
            && typeof selector.variant_id === 'string'
            && typeof selector.text === 'string'
        ))
}

function readMetaTimestamp (value: Record<string, unknown>, key: string): string|null {
    if (!isRecord(value.meta)) {
        return null
    }

    return typeof value.meta[key] === 'string' ? value.meta[key] : null
}

function isRecord (value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readNonNegativeNumber (value: unknown, fallback: number): number {
    return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : fallback
}
