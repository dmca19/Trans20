// LangChain tool schemas intentionally use Zod v3; do not switch this to the package default v4 import.
import { z } from 'zod/v3'

import {
    type AppliesTo,
    type EntryStatus,
    type EntryType,
    type GenderPresentation,
    type GlossaryEntryApplicability,
    type GlossaryEntryContent,
    type GlossaryEntryPolicy,
    type GlossaryEntryTarget,
    type MergeProposalStatus,
    type PolicyStrength,
    type RejectedReason,
    type ReviewEntryBatchOperation,
    type ReviewTermOperation,
    type TermStatus,
    type TermType,
    type TranslationRuleBasis,
    appliesToValues,
    entryStatusValues,
    entryTypeValues,
    genderPresentationConfidenceValues,
    genderPresentationValueValues,
    mergeProposalStatusValues,
    policyStrengthValues,
    rejectedReasonValues,
    reviewEntryBatchOperationValues,
    termStatusValues,
    termTypeValues,
    translationRuleBasisValues,
} from './glossaryStore.js'
import { DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE } from './defaultTranslationLanguages.js'

export const MAX_QUERY_LIMIT = 100
export const MAX_BATCH_TOOL_ITEMS = 15
export const MAX_ENTRY_SEARCH_LIMIT = 100
export const MAX_EVIDENCE_CONTEXT_WINDOW = 500
export const MAX_CHARS_PER_KEY = 2000
export const DEFAULT_EVIDENCE_CONTEXT_WINDOW = 20
export const DEFAULT_MAX_CHARS_PER_KEY = 500
export type ValidationError = {
    field: string
    message: string
    code?: string
}

export type SubmitGlossaryPlanOutput = {
    ok: true
    plan_id: string
    retry_required: false
}|{
    ok: false
    errors: ValidationError[]
    retry_required: true
}

export type MatchMode = 'exact'|'alias'|'compound'|'fuzzy'|'regex'
export type MetadataOperation = 'add_aliases'|'change_term_type'|'deprecate_term'|'reject_term'|'merge_term'|'append_gender_presentation'
export type EntryOperation = 'append_entry'|'modify_entry'
export type { TranslationRuleBasis }
export type GenderPresentationPayload = Omit<GenderPresentation, 'entry_id'>

export type SessionSnapshot = {
    dataRevision: string
}

export type EvidenceRange = {
    source_file_id: string
    start_index: number
    end_index: number
    allowed_key_indices?: number[]
}

export type EvidenceScope = {
    source_file_id: string
    batch_start_index: number
    batch_end_index: number
    context_window: number
    allowed_key_indices?: number[]
    filtered_index_to_key_index?: Record<number, number>
    key_index_to_filtered_index?: Record<number, number>
}

export type GlossaryToolSession = {
    snapshots: SessionSnapshot[]
    queriedTermIds: Set<string>
    queriedTermRevisions: Map<string, number>
    queriedSourceKeys: Map<string, Set<MatchMode>>
    queriedMergeProposalPairs: Set<string>
    returnedEvidenceIds: Set<string>
    allowedEvidenceRanges: EvidenceRange[]
    listedReviewWindowIds: Set<string>
    checkedCoverageReviewWindowIds: Set<string>
}

export type ManualTransData = {
    relativePath: string
    keys: string[]
    values: unknown[]
}

export type QueryGlossaryTermsInput = {
    queries: {
        text: string
        match_modes: MatchMode[]
        case_sensitive: boolean
    }[]
    source_language: string
    include_entries: boolean
    include_evidence: boolean
    include_rejected_details: boolean
    entry_statuses: EntryStatus[]|null
    term_statuses: TermStatus[]|null
    limit: number
}

export type SearchGlossaryEntriesInput = {
    entry_filter: EntryFilter|null
    term_filter: TermFilter|null
    include_term: boolean
    include_evidence: boolean
    limit: number
}

export type EntryFilter = {
    entry_types: EntryType[]|null
    entry_statuses: EntryStatus[]|null
    applies_to: AppliesTo[]|null
    target_language: string|null
    policy_strength: PolicyStrength[]|null
    domain: string|null
    text: string|null
    text_fields: EntryTextField[]|null
    is_regex: boolean
    case_sensitive: boolean
}

export type TermFilter = {
    source_language: string|null
    term_types: TermType[]|null
    term_statuses: TermStatus[]|null
}

export type EntryTextField =
    'content.summary'
    |'content.description'
    |'target.preferred_translation'
    |'target.alternative_translations'
    |'target.forbidden_translations'
    |'policy.notes'
    |'applicability.applies_when'
    |'applicability.does_not_apply_when'

export type EvidenceContextInput = {
    evidence_ids: string[]
    window_before: number
    window_after: number
    max_chars_per_key: number
    include_values: boolean
}

export type CreateOrGetTermInput = {
    source_text: string
    source_language: string
    term_type: TermType
    aliases: string[]
    status: 'active'
    confirmed_distinct_from_term_ids: string[]|null
    distinct_reason: string|null
    created_by: string
    created_from: Record<string, unknown>
}

export type CreateOrGetTermsInput = {
    items: (CreateOrGetTermInput & {
        client_id: string
    })[]
}

export type QueryTermMergeProposalsInput = {
    source_term_id: string|null
    target_term_id: string|null
    statuses: MergeProposalStatus[]|null
    include_terms: boolean
    limit: number
}

export type CreateTermMergeProposalInput = {
    source_term_id: string
    target_term_id: string
    status: 'candidate'
    reason: string
    evidence: EvidencePayload[]
    existing_evidence_ids: string[]
    created_by: string
    created_from: Record<string, unknown>
}

export type UpdateTermMetadataInput = {
    operation: MetadataOperation
    term_id: string
    expected_term_revision: number
    aliases_to_add: string[]|null
    term_type: TermType|null
    merged_into: string|null
    gender_presentation: GenderPresentation|null
    rejected_reason: RejectedReason|null
    reason: string
    updated_by: string
    updated_from: Record<string, unknown>
}

export type UpdateTermEntriesInput = {
    operation: EntryOperation
    term_id: string
    entry_id: string|null
    expected_entry_revision: number|null
    entry: EntryPayload
    gender_presentation: GenderPresentationPayload|null
    evidence: EvidencePayload[]
    created_by: string
    created_from: Record<string, unknown>
}

export type AppendTermEntriesBatchInput = {
    items: (UpdateTermEntriesInput & {
        client_id: string
    })[]
}

export type ListReviewCandidatesInput = {
    review_window_id: string|null
    include_terms: boolean
    include_entries: boolean
    include_evidence: boolean
    limit: number
}

export type CheckTranslationRuleCoverageInput = {
    review_window_id: string
    term_ids: string[]|null
    term_types: TermType[]|null
    include_rejected_candidates: boolean
}

export type AppendReviewEntryInput = {
    review_window_id: string
    term_id: string
    entry_type: EntryType
    summary: string
    description: string
    applies_to: AppliesTo
    applies_when: string[]|null
    does_not_apply_when: string[]|null
    policy_strength: PolicyStrength
    requires_context_check: boolean|null
    notes: string[]|null
    basis: TranslationRuleBasis|null
    source_variant_indexes: number[]|null
    preferred_translation: string|null
    alternative_translations: string[]|null
    forbidden_translations: string[]|null
    evidence_ids: string[]
    evidence: EvidencePayload[]
    reason: string
}

export type ReviewEntriesBatchInput = {
    review_window_id: string
    actions: {
        operation: ReviewEntryBatchOperation
        entry_id: string
        expected_entry_revision?: number|null
        target_entry_id?: string|null
        target_term_id?: string|null
        revised_entry?: Partial<Pick<EntryPayload, 'entry_type'|'basis'|'content'|'target'|'applicability'|'policy'>>|null
        reason: string
    }[]
}

export type ReviewTermsBatchInput = {
    review_window_id: string
    actions: {
        operation: ReviewTermOperation
        term_id: string
        expected_term_revision?: number|null
        target_term_id?: string|null
        term_type?: TermType|null
        aliases_to_add?: string[]|null
        aliases_to_remove?: string[]|null
        entry_ids?: string[]|null
        gender_presentations?: GenderPresentation[]|null
        rejected_reason?: RejectedReason|null
        reason: string
    }[]
}

export type SubmitGlossaryWorkerBatchInput = {
    // batch_id and batch_number are intentionally redundant. The worker submit
    // tool checks both to catch stale context, slot mix-ups, or misrouted runs.
    batch_id: string
    batch_number: number
    summary: string
    deferred_notes: string[]
}

export type SubmitGlossaryReviewInput = {
    review_window_id: string
    summary: string
    deferred_notes: string[]
}

export type EntryPayload = {
    entry_type: EntryType
    basis?: TranslationRuleBasis
    content: GlossaryEntryContent
    target?: GlossaryEntryTarget
    applicability: GlossaryEntryApplicability & {
        source_variant_indexes?: number[]
    }
    policy: GlossaryEntryPolicy
    status: EntryStatus
}

export type EvidencePayload = {
    source_ref: {
        source_file_id: string
        filtered_key_index: number
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
}

const nonEmptyString = z.string().trim().min(1)
const nullableString = z.union([nonEmptyString, z.null()]).optional()
const recordSchema = z.record(z.unknown())

const matchModeSchema = z.enum(['exact', 'alias', 'compound', 'fuzzy', 'regex'])
const termTypeSchema = z.enum(termTypeValues)
const termStatusSchema = z.enum(termStatusValues)
const rejectedReasonSchema = z.enum(rejectedReasonValues)
const mergeProposalStatusSchema = z.enum(mergeProposalStatusValues)
const entryTypeSchema = z.enum(entryTypeValues)
const entryStatusSchema = z.enum(entryStatusValues)
const policyStrengthSchema = z.enum(policyStrengthValues)
const appliesToSchema = z.enum(appliesToValues)
const genderPresentationValueSchema = z.enum(genderPresentationValueValues)
const genderPresentationConfidenceSchema = z.enum(genderPresentationConfidenceValues)
const translationRuleBasisSchema = z.enum(translationRuleBasisValues)

const genderPresentationPayloadSchema = z.object({
    value: genderPresentationValueSchema,
    confidence: genderPresentationConfidenceSchema,
}).strict()

const genderPresentationSchema = genderPresentationPayloadSchema.extend({
    entry_id: nonEmptyString,
}).strict()

export const glossaryPlanSchema = z.object({
    term_extraction_policy: z.object({
        term_types: z.array(termTypeSchema).nonempty(),
        entry_types: z.array(entryTypeSchema).nonempty(),
        default_term_status: z.literal('active'),
        default_entry_status: z.literal('candidate'),
        require_evidence: z.literal(true),
        one_entry_one_claim: z.literal(true),
    }).strict(),
    shared_prompt_context: z.object({
        task: nonEmptyString,
        source_language: z.literal(DEFAULT_SOURCE_LANGUAGE),
        target_language: z.literal(DEFAULT_TARGET_LANGUAGE),
        project_context: nonEmptyString,
        domain_hint: nonEmptyString,
        focus: z.array(nonEmptyString).nonempty(),
        cautions: z.array(nonEmptyString).nonempty(),
        notes: z.array(nonEmptyString).nonempty(),
    }).strict(),
}).strict()

export const queryGlossaryTermsSchema = z.object({
    queries: z.array(z.object({
        text: nonEmptyString,
        match_modes: z.array(matchModeSchema).nonempty(),
        case_sensitive: z.boolean().optional().default(false),
    }).strict()).nonempty(),
    source_language: nonEmptyString,
    include_entries: z.boolean().optional().default(false),
    include_evidence: z.boolean().optional().default(false),
    include_rejected_details: z.boolean().optional().default(false),
    entry_statuses: z.array(entryStatusSchema).nullable().optional(),
    term_statuses: z.array(termStatusSchema).nullable().optional(),
    limit: z.number().int().min(1).max(MAX_QUERY_LIMIT),
}).strict()

export const reviewQueryGlossaryTermsSchema = queryGlossaryTermsSchema.omit({
    include_rejected_details: true,
})

const entryTextFieldSchema = z.enum([
    'content.summary',
    'content.description',
    'target.preferred_translation',
    'target.alternative_translations',
    'target.forbidden_translations',
    'policy.notes',
    'applicability.applies_when',
    'applicability.does_not_apply_when',
])

export const searchGlossaryEntriesSchema = z.object({
    entry_filter: z.object({
        entry_types: z.array(entryTypeSchema).nullable().optional(),
        entry_statuses: z.array(entryStatusSchema).nullable().optional(),
        applies_to: z.array(appliesToSchema).nullable().optional(),
        target_language: nullableString,
        policy_strength: z.array(policyStrengthSchema).nullable().optional(),
        domain: nullableString,
        text: nullableString,
        text_fields: z.array(entryTextFieldSchema).nullable().optional(),
        is_regex: z.boolean().optional().default(false),
        case_sensitive: z.boolean().optional().default(false),
    }).strict().nullable().optional(),
    term_filter: z.object({
        source_language: nullableString,
        term_types: z.array(termTypeSchema).nullable().optional(),
        term_statuses: z.array(termStatusSchema).nullable().optional(),
    }).strict().nullable().optional(),
    include_term: z.boolean().optional().default(false),
    include_evidence: z.boolean().optional().default(false),
    limit: z.number().int().min(1).max(MAX_ENTRY_SEARCH_LIMIT),
}).strict()

export const createOrGetTermSchema = z.object({
    source_text: nonEmptyString,
    source_language: nonEmptyString,
    term_type: termTypeSchema,
    aliases: z.array(nonEmptyString),
    status: z.literal('active'),
    confirmed_distinct_from_term_ids: z.array(nonEmptyString).nullable().optional(),
    distinct_reason: nonEmptyString.nullable().optional(),
    created_by: nonEmptyString,
    created_from: recordSchema,
}).strict()

export const createOrGetTermsSchema = z.object({
    items: z.array(createOrGetTermSchema.extend({
        client_id: nonEmptyString,
    }).strict()).nonempty().max(MAX_BATCH_TOOL_ITEMS),
}).strict()

export const queryTermMergeProposalsSchema = z.object({
    source_term_id: nonEmptyString.nullable().optional(),
    target_term_id: nonEmptyString.nullable().optional(),
    statuses: z.array(mergeProposalStatusSchema).nullable().optional(),
    include_terms: z.boolean().optional().default(false),
    limit: z.number().int().min(1).max(MAX_QUERY_LIMIT),
}).strict()

export const updateTermMetadataSchema = z.object({
    operation: z.enum(['add_aliases', 'change_term_type', 'deprecate_term', 'reject_term', 'merge_term', 'append_gender_presentation']),
    term_id: nonEmptyString,
    expected_term_revision: z.number().int().positive(),
    aliases_to_add: z.array(nonEmptyString).nullable().optional(),
    term_type: termTypeSchema.nullable().optional(),
    merged_into: nonEmptyString.nullable().optional(),
    gender_presentation: genderPresentationSchema.nullable().optional(),
    rejected_reason: rejectedReasonSchema.nullable().optional(),
    reason: nonEmptyString,
    updated_by: nonEmptyString,
    updated_from: recordSchema,
}).strict()

const targetSchema = z.object({
    target_language: nonEmptyString,
    preferred_translation: nonEmptyString.optional(),
    alternative_translations: z.array(nonEmptyString).optional(),
    forbidden_translations: z.array(nonEmptyString).optional(),
}).strict()

const entryContentSchema = z.object({
    summary: nonEmptyString,
    description: nonEmptyString,
}).strict()

const entryApplicabilitySchema = z.object({
    applies_to: appliesToSchema,
    domain: nonEmptyString.optional(),
    applies_when: z.array(nonEmptyString).optional(),
    does_not_apply_when: z.array(nonEmptyString).optional(),
    source_variant_indexes: z.array(z.number().int().min(0)).optional(),
}).strict()

const entryPolicySchema = z.object({
    strength: policyStrengthSchema,
    requires_context_check: z.boolean().optional(),
    notes: z.array(nonEmptyString).optional(),
}).strict()

const looseEntryPayloadSchema = z.object({
    entry_type: entryTypeSchema,
    basis: translationRuleBasisSchema.optional(),
    content: entryContentSchema,
    target: targetSchema.optional(),
    applicability: entryApplicabilitySchema,
    policy: entryPolicySchema,
    status: z.literal('candidate'),
}).strict()

const entryPayloadSchema = looseEntryPayloadSchema.superRefine((entry, ctx) => {
    if ((entry.entry_type === 'fact' || entry.entry_type === 'style' || entry.entry_type === 'continuity') && entry.target) {
        ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ['target'],
            message: `${entry.entry_type} Entry 不允许包含 target。`,
        })
    }

    if (entry.entry_type === 'translation_rule' && !entry.basis) {
        ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ['basis'],
            message: 'translation_rule 必须提供 basis。',
        })
    }

    if (entry.entry_type === 'translation_rule' && !hasTargetConstraint(entry.target)) {
        ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ['target'],
            message: 'translation_rule 必须提供 target，且至少包含 preferred_translation、alternative_translations 或 forbidden_translations。',
        })
    }

    if (entry.entry_type !== 'translation_rule' && entry.basis) {
        ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ['basis'],
            message: 'basis 只允许用于 translation_rule Entry。',
        })
    }

    if (entry.entry_type === 'translation_rule' && entry.target && entry.target.target_language !== DEFAULT_TARGET_LANGUAGE) {
        ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ['target', 'target_language'],
            message: `target_language 必须为 ${DEFAULT_TARGET_LANGUAGE}。`,
        })
    }

    if (entry.entry_type === 'translation_rule' && (!entry.applicability.source_variant_indexes || entry.applicability.source_variant_indexes.length === 0)) {
        ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ['applicability', 'source_variant_indexes'],
            message: 'translation_rule 必须提供 applicability.source_variant_indexes。',
        })
    }

    if (entry.entry_type !== 'translation_rule' && entry.applicability.source_variant_indexes) {
        ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ['applicability', 'source_variant_indexes'],
            message: '非 translation_rule Entry 不允许包含 source selector 字段。',
        })
    }
})

function hasTargetConstraint (target: z.infer<typeof targetSchema>|undefined): boolean {
    if (!target) {
        return false
    }

    return Boolean(target.preferred_translation)
        || Boolean(target.alternative_translations?.length)
        || Boolean(target.forbidden_translations?.length)
}

const evidencePayloadSchema = z.object({
    source_ref: z.object({
        source_file_id: nonEmptyString,
        filtered_key_index: z.number().int().min(0),
        key_hash: nonEmptyString.optional(),
        span: z.object({
            start: z.number().int().min(0),
            end: z.number().int().min(0),
        }).strict().optional(),
    }).strict(),
    quote: nonEmptyString,
    context: z.object({
        before: z.union([nonEmptyString, z.null()]).optional(),
        after: z.union([nonEmptyString, z.null()]).optional(),
    }).strict().optional(),
    reason: nonEmptyString,
}).strict()

export const createTermMergeProposalSchema = z.object({
    source_term_id: nonEmptyString,
    target_term_id: nonEmptyString,
    status: z.literal('candidate'),
    reason: nonEmptyString,
    evidence: z.array(evidencePayloadSchema).optional().default([]),
    existing_evidence_ids: z.array(nonEmptyString).optional().default([]),
    created_by: nonEmptyString,
    created_from: recordSchema,
}).strict()

export const updateTermEntriesSchema = z.object({
    operation: z.enum(['append_entry', 'modify_entry']),
    term_id: nonEmptyString,
    entry_id: nonEmptyString.nullable().optional(),
    expected_entry_revision: z.number().int().positive().nullable().optional(),
    entry: entryPayloadSchema,
    gender_presentation: genderPresentationPayloadSchema.nullable().optional(),
    evidence: z.array(evidencePayloadSchema).optional().default([]),
    created_by: nonEmptyString,
    created_from: recordSchema,
}).strict()

const looseUpdateTermEntriesSchemaBase = updateTermEntriesSchema.extend({
    entry: looseEntryPayloadSchema,
}).strict()

export const appendTermEntriesBatchSchema = z.object({
    items: z.array(updateTermEntriesSchema.extend({
        client_id: nonEmptyString,
        operation: z.literal('append_entry').optional().default('append_entry'),
    }).strict()).nonempty().max(MAX_BATCH_TOOL_ITEMS),
}).strict()

const looseAppendTermEntriesBatchSchemaBase = z.object({
    items: z.array(looseUpdateTermEntriesSchemaBase.extend({
        client_id: nonEmptyString,
        operation: z.literal('append_entry').optional().default('append_entry'),
    }).strict()).nonempty().max(MAX_BATCH_TOOL_ITEMS),
}).strict()

export const listReviewCandidatesSchema = z.object({
    review_window_id: nonEmptyString.nullable().optional(),
    include_terms: z.boolean().optional().default(true),
    include_entries: z.boolean().optional().default(true),
    include_evidence: z.boolean().optional().default(true),
    limit: z.number().int().min(1).max(MAX_QUERY_LIMIT).optional().default(MAX_QUERY_LIMIT),
}).strict()

export const checkTranslationRuleCoverageSchema = z.object({
    review_window_id: nonEmptyString,
    term_ids: z.array(nonEmptyString).nullable().optional(),
    term_types: z.array(termTypeSchema).nullable().optional(),
    include_rejected_candidates: z.boolean().optional().default(true),
}).strict()

export const appendReviewEntrySchema = z.object({
    review_window_id: nonEmptyString,
    term_id: nonEmptyString,
    entry_type: entryTypeSchema,
    summary: nonEmptyString,
    description: nonEmptyString,
    applies_to: appliesToSchema.optional().default('term'),
    applies_when: z.array(nonEmptyString).nullable().optional(),
    does_not_apply_when: z.array(nonEmptyString).nullable().optional(),
    policy_strength: policyStrengthSchema.optional().default('recommended'),
    requires_context_check: z.boolean().nullable().optional(),
    notes: z.array(nonEmptyString).nullable().optional(),
    basis: translationRuleBasisSchema.nullable().optional(),
    source_variant_indexes: z.array(z.number().int().min(0)).nullable().optional(),
    preferred_translation: nonEmptyString.nullable().optional(),
    alternative_translations: z.array(nonEmptyString).nullable().optional(),
    forbidden_translations: z.array(nonEmptyString).nullable().optional(),
    evidence_ids: z.array(nonEmptyString).optional().default([]),
    evidence: z.array(evidencePayloadSchema).optional().default([]),
    reason: nonEmptyString,
}).strict()

const revisedEntrySchema = z.object({
    entry_type: entryTypeSchema.optional(),
    basis: translationRuleBasisSchema.optional(),
    content: z.object({
        summary: nonEmptyString,
        description: nonEmptyString,
    }).strict().optional(),
    target: targetSchema.optional(),
    applicability: z.object({
        applies_to: appliesToSchema,
        domain: nonEmptyString.optional(),
        applies_when: z.array(nonEmptyString).optional(),
        does_not_apply_when: z.array(nonEmptyString).optional(),
        source_variant_indexes: z.array(z.number().int().min(0)).optional(),
    }).strict().optional(),
    policy: z.object({
        strength: policyStrengthSchema,
        requires_context_check: z.boolean().optional(),
        notes: z.array(nonEmptyString).optional(),
    }).strict().optional(),
}).strict()

export const reviewEntriesBatchSchema = z.object({
    review_window_id: nonEmptyString,
    actions: z.array(z.object({
        operation: z.enum(reviewEntryBatchOperationValues),
        entry_id: nonEmptyString,
        expected_entry_revision: z.number().int().positive().nullable().optional(),
        target_entry_id: nonEmptyString.nullable().optional(),
        target_term_id: nonEmptyString.nullable().optional(),
        revised_entry: revisedEntrySchema.nullable().optional(),
        reason: nonEmptyString,
    }).strict()).nonempty().max(MAX_BATCH_TOOL_ITEMS),
}).strict()

export const reviewTermsBatchSchema = z.object({
    review_window_id: nonEmptyString,
    actions: z.array(z.object({
        operation: z.enum(['keep_active', 'reject', 'deprecate', 'change_term_type', 'add_aliases', 'remove_invalid_aliases', 'merge_term', 'move_entries', 'set_gender_presentations']),
        term_id: nonEmptyString,
        expected_term_revision: z.number().int().positive().nullable().optional(),
        target_term_id: nonEmptyString.nullable().optional(),
        term_type: termTypeSchema.nullable().optional(),
        aliases_to_add: z.array(nonEmptyString).nullable().optional(),
        aliases_to_remove: z.array(nonEmptyString).nullable().optional(),
        entry_ids: z.array(nonEmptyString).nullable().optional(),
        gender_presentations: z.array(genderPresentationSchema).nullable().optional(),
        rejected_reason: rejectedReasonSchema.nullable().optional(),
        reason: nonEmptyString,
    }).strict()).nonempty().max(MAX_BATCH_TOOL_ITEMS),
}).strict()

export const submitGlossaryWorkerBatchSchema = z.object({
    batch_id: nonEmptyString,
    batch_number: z.number().int().min(1),
    summary: nonEmptyString,
    deferred_notes: z.array(nonEmptyString),
}).strict()

export const submitGlossaryReviewSchema = z.object({
    review_window_id: nonEmptyString,
    summary: nonEmptyString,
    deferred_notes: z.array(nonEmptyString),
}).strict()

export const looseGlossaryPlanSchema = glossaryPlanSchema
export const looseQueryGlossaryTermsSchema = queryGlossaryTermsSchema
export const looseReviewQueryGlossaryTermsSchema = reviewQueryGlossaryTermsSchema
export const looseSearchGlossaryEntriesSchema = searchGlossaryEntriesSchema
export const looseEvidenceContextSchema = z.object({
    evidence_ids: z.array(nonEmptyString).nonempty(),
    window_before: z.number().int().min(0).optional().default(DEFAULT_EVIDENCE_CONTEXT_WINDOW),
    window_after: z.number().int().min(0).optional().default(DEFAULT_EVIDENCE_CONTEXT_WINDOW),
    max_chars_per_key: z.number().int().min(1).optional().default(DEFAULT_MAX_CHARS_PER_KEY),
    include_values: z.boolean().optional().default(false),
}).strict()
export const looseCreateOrGetTermSchema = createOrGetTermSchema
export const looseCreateOrGetTermsSchema = createOrGetTermsSchema
export const looseQueryTermMergeProposalsSchema = queryTermMergeProposalsSchema
export const looseCreateTermMergeProposalSchema = createTermMergeProposalSchema
export const looseUpdateTermMetadataSchema = updateTermMetadataSchema
export const looseUpdateTermEntriesSchema = looseUpdateTermEntriesSchemaBase
export const looseAppendTermEntriesBatchSchema = looseAppendTermEntriesBatchSchemaBase
export const looseListReviewCandidatesSchema = listReviewCandidatesSchema
export const looseCheckTranslationRuleCoverageSchema = checkTranslationRuleCoverageSchema
export const looseAppendReviewEntrySchema = appendReviewEntrySchema
export const looseReviewEntriesBatchSchema = reviewEntriesBatchSchema
export const looseReviewTermsBatchSchema = reviewTermsBatchSchema
export const looseSubmitGlossaryWorkerBatchSchema = submitGlossaryWorkerBatchSchema
export const looseSubmitGlossaryReviewSchema = submitGlossaryReviewSchema
