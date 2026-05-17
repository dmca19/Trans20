import { createHash } from 'node:crypto'

import { tool, type StructuredToolInterface } from '@langchain/core/tools'
import { distance as levenshteinDistance } from 'fastest-levenshtein'
// LangChain tool schema parsing intentionally uses Zod v3 types to match the schemas passed to LangChain tools.
import type { ZodTypeAny } from 'zod/v3'

import { collectCharacterContextWarnings } from './characterContextWarnings.js'
import { createYieldController } from './eventLoop.js'
import {
    appendGlossaryPlan,
    entryTypeValues,
    type EntryStatus,
    type GenderPresentation,
    type GlossaryEntry,
    type GlossaryEvidence,
    type GlossaryMergeProposal,
    type GlossaryPlan,
    type GlossaryPlanInput,
    type GlossaryReviewWindow,
    type GlossaryState,
    type GlossaryTerm,
    readGlossaryState,
    type RejectedReason,
    type ReviewEntryAction,
    type ReviewTermAction,
    type TermStatus,
    updateGlossaryState,
} from './glossaryStore.js'
import {
    addAliasesToTerm,
    createAliasState,
    ensureAliasForText,
    entryReferencesAliasId,
    findAliasIdByText,
    formatEntryForAgent,
    formatTermForAgent,
    getAliasTexts,
    getSourceVariants,
    removeAliasesFromTermByText,
    resolveSourceVariantIndexes,
} from './glossaryAliasUtils.js'
import { collectTranslationRuleCoverageIssues } from './glossaryQuality.js'
import { filterManualTransKeyItems } from './keyFilter.js'
import { DEFAULT_SOURCE_LANGUAGE, DEFAULT_TARGET_LANGUAGE } from './defaultTranslationLanguages.js'
import { normalizeProjectFileId, resolveProjectFile } from './pathUtils.js'
import { readJsonFile } from './fileUtils.js'
import { jsonOutput, truncate, withToolLogging, type ToolCallLogger } from './toolRuntime.js'
import {
    type AppendReviewEntryInput,
    type AppendTermEntriesBatchInput,
    type CheckTranslationRuleCoverageInput,
    type CreateOrGetTermInput,
    type CreateOrGetTermsInput,
    type CreateTermMergeProposalInput,
    type EntryFilter,
    type EntryPayload,
    type EntryTextField,
    type EvidenceContextInput,
    type EvidencePayload,
    type EvidenceRange,
    type EvidenceScope,
    type GlossaryToolSession,
    type ListReviewCandidatesInput,
    type ManualTransData,
    type MatchMode,
    MAX_BATCH_TOOL_ITEMS,
    MAX_CHARS_PER_KEY,
    MAX_EVIDENCE_CONTEXT_WINDOW,
    type QueryGlossaryTermsInput,
    type QueryTermMergeProposalsInput,
    type ReviewEntriesBatchInput,
    type ReviewTermsBatchInput,
    type SearchGlossaryEntriesInput,
    type SubmitGlossaryPlanOutput,
    type SubmitGlossaryReviewInput,
    type SubmitGlossaryWorkerBatchInput,
    type TermFilter,
    type UpdateTermEntriesInput,
    type UpdateTermMetadataInput,
    type ValidationError,
    appendTermEntriesBatchSchema,
    looseAppendTermEntriesBatchSchema,
    looseCreateOrGetTermSchema,
    looseCreateOrGetTermsSchema,
    looseCreateTermMergeProposalSchema,
    looseEvidenceContextSchema,
    looseGlossaryPlanSchema,
    looseQueryTermMergeProposalsSchema,
    looseQueryGlossaryTermsSchema,
    looseReviewQueryGlossaryTermsSchema,
    looseListReviewCandidatesSchema,
    looseCheckTranslationRuleCoverageSchema,
    looseAppendReviewEntrySchema,
    looseSearchGlossaryEntriesSchema,
    looseReviewEntriesBatchSchema,
    looseReviewTermsBatchSchema,
    looseSubmitGlossaryReviewSchema,
    looseSubmitGlossaryWorkerBatchSchema,
    looseUpdateTermEntriesSchema,
    looseUpdateTermMetadataSchema,
    updateTermEntriesSchema,
} from './glossaryToolTypes.js'

type QueryGlossaryTermsOutput = {
    ok: true
    matched_terms: {
        term: GlossaryTerm
        match: {
            query_text: string
            match_mode: MatchMode
            matched_text: string
        }
        entries: GlossaryEntry[]
        evidence: GlossaryEvidence[]
    }[]
    warnings: string[]
}

type ValidatedQueryGlossaryTermsInput = QueryGlossaryTermsInput & {
    include_rejected_details: boolean
}

type NormalizedEvidencePayload = Omit<EvidencePayload, 'source_ref'> & {
    source_ref: GlossaryEvidence['source_ref'] & {
        source_file_id: string
        key_index: number
    }
}

export type CreateGlossaryToolsOptions = {
    manualTransFile: string
    projectRoot?: string
    onPlanSubmitted?: (plan: GlossaryPlan) => void
    onWorkerBatchSubmitted?: (submission: SubmitGlossaryWorkerBatchInput) => void
    onReviewSubmitted?: (submission: SubmitGlossaryReviewInput) => void
    evidenceScope?: EvidenceScope
    enableBatchTools?: boolean
    enableTargetTermRevisionCheck?: boolean
    enableCharacterGenderWarnings?: boolean
    reviewMode?: boolean
    readOnly?: boolean
    createdFromBatchId?: string
    workerBatchId?: string
    workerBatchNumber?: number
    summaryMaxChars?: number|null
    deferredNotesMaxItems?: number|null
    requireWorkerBatchRead?: () => boolean
}

export type GlossaryReviewCommitResult = {
    review_window_id: string
    entry_action_count: number
    term_action_count: number
    cache_applied_count: number
    cache_conflict_count: number
    warnings?: unknown[]
    character_context_warnings?: unknown[]
}

export function createGlossaryTools (
    root: string,
    onToolEvent: ToolCallLogger,
    manualTransFileOrPlanSubmitted: string|CreateGlossaryToolsOptions,
    onPlanSubmitted?: (plan: GlossaryPlan) => void,
): StructuredToolInterface[] {
    const options = normalizeGlossaryToolsOptions(root, manualTransFileOrPlanSubmitted, onPlanSubmitted)
    const manualTransFile = options.manualTransFile
    const projectRoot = options.projectRoot
    const planSubmitted = options.onPlanSubmitted
    let workerBatchSubmitted = false
    let reviewSubmitted = false
    const session: GlossaryToolSession = {
        snapshots: [],
        queriedTermIds: new Set<string>(),
        queriedTermRevisions: new Map<string, number>(),
        queriedSourceKeys: new Map<string, Set<MatchMode>>(),
        queriedMergeProposalPairs: new Set<string>(),
        returnedEvidenceIds: new Set<string>(),
        allowedEvidenceRanges: [],
        listedReviewWindowIds: new Set<string>(),
        checkedCoverageReviewWindowIds: new Set<string>(),
    }
    const evidenceScope = options.evidenceScope
    const queryGlossaryTermsSchema = options.reviewMode === true
        ? looseReviewQueryGlossaryTermsSchema
        : looseQueryGlossaryTermsSchema
    const queryGlossaryTermsDescription = options.reviewMode === true
        ? 'Query glossary Terms by source text, aliases, compound text, fuzzy match, or regex. Defaults to all Term statuses and returns rejected Term entries/evidence in review mode. Use before reviewing or writing Terms, Entries, or Evidence.'
        : 'Query glossary Terms by source text, aliases, compound text, fuzzy match, or regex. Defaults to all Term statuses. Rejected Terms are visible by default with rejected_reason but without entries/evidence; pass include_rejected_details: true only when you need rejected entries/evidence. Use before writing Terms, Entries, or Evidence.'

    const tools = [
        tool(async input => withToolLogging('query_glossary_terms', input, onToolEvent, async () => {
            const validation = validateQueryGlossaryTermsInput(input)

            if (!validation.ok) {
                return jsonOutput({
                    ok: false,
                    code: 'invalid_query',
                    errors: validation.errors,
                })
            }

            const value: ValidatedQueryGlossaryTermsInput = {
                ...validation.value,
                include_rejected_details: validation.value.include_rejected_details || options.reviewMode === true,
            }

            const output = await readGlossaryState(root, async state => {
                const result = await runQueryGlossaryTerms(state, value)

                recordSnapshot(session)
                for (const query of value.queries) {
                    recordQueriedSourceKey(session, query.text, value.source_language, query.match_modes)
                }
                for (const match of result.matched_terms) {
                    recordQueriedTerm(session, match.term)
                    for (const evidence of match.evidence) {
                        session.returnedEvidenceIds.add(evidence.evidence_id)
                    }
                }

                return result
            })

            return jsonOutput(await formatQueryGlossaryTermsOutput(projectRoot, manualTransFile, output))
        }), {
            name: 'query_glossary_terms',
            description: queryGlossaryTermsDescription,
            schema: queryGlossaryTermsSchema,
        }),
        tool(async input => withToolLogging('search_glossary_entries', input, onToolEvent, async () => {
            const validation = validateSearchGlossaryEntriesInput(input)

            if (!validation.ok) {
                return jsonOutput({
                    ok: false,
                    code: 'invalid_query',
                    errors: validation.errors,
                })
            }

            const output = await readGlossaryState(root, async state => {
                const result = await runSearchGlossaryEntries(state, validation.value)

                recordSnapshot(session)
                for (const match of result.matched_entries) {
                    if (options.reviewMode) {
                        recordQueriedTerm(session, match.term)
                    }
                    for (const evidence of match.evidence) {
                        session.returnedEvidenceIds.add(evidence.evidence_id)
                    }
                }

                return result
            })

            return jsonOutput(await formatSearchGlossaryEntriesOutput(projectRoot, manualTransFile, output))
        }), {
            name: 'search_glossary_entries',
            description: 'Search existing glossary Entries by structured filters or text fields and optionally return related Terms and Evidence.',
            schema: looseSearchGlossaryEntriesSchema,
        }),
        tool(async input => withToolLogging('get_evidence_context', input, onToolEvent, async () => {
            const validation = validateEvidenceContextInput(input)

            if (!validation.ok) {
                return jsonOutput({
                    ok: false,
                    code: evidenceContextValidationCode(validation.errors),
                    errors: validation.errors,
                })
            }

            const output = await readGlossaryState(root, state => runGetEvidenceContext(projectRoot, manualTransFile, state, session, validation.value))

            return jsonOutput(output)
        }), {
            name: 'get_evidence_context',
            description: 'Read source-key context around Evidence ids returned by glossary query/search tools. Does not accept arbitrary key ranges.',
            schema: looseEvidenceContextSchema,
        }),
        ...(options.reviewMode ? [
            tool(async input => withToolLogging('list_review_candidates', input, onToolEvent, async () => {
                const validation = validateListReviewCandidatesInput(input)

                if (!validation.ok) {
                    return jsonOutput(validationErrorOutput(validation.errors))
                }

                const output = await readGlossaryState(root, state => runListReviewCandidates(projectRoot, manualTransFile, state, validation.value, options.enableCharacterGenderWarnings))
                if (isRecord(output) && output.ok === true && typeof output.review_window_id === 'string') {
                    session.listedReviewWindowIds.add(output.review_window_id)
                }
                recordReviewCandidateEvidence(session, output)
                return jsonOutput(await formatReviewCandidatesOutput(projectRoot, manualTransFile, output))
            }), {
                name: 'list_review_candidates',
                description: 'List the current glossary review window, unreviewed batches, new Terms/Entries/Evidence, structural issues, conflicts, and merge proposals.',
                schema: looseListReviewCandidatesSchema,
            }),
            tool(async input => withToolLogging('check_translation_rule_coverage', input, onToolEvent, async () => {
                const validation = validateCheckTranslationRuleCoverageInput(input)

                if (!validation.ok) {
                    return jsonOutput(validationErrorOutput(validation.errors))
                }

                const output = await readGlossaryState(root, state => {
                    const window = state.review.active_window

                    if (!window || window.review_window_id !== validation.value.review_window_id) {
                        return {
                            ok: false,
                            code: 'no_active_review_window',
                            message: `当前没有 active review window ${validation.value.review_window_id}。`,
                            retry_required: false,
                        }
                    }

                    const output = collectTranslationRuleCoverageIssues(state, validation.value)
                    if (isRecord(output) && output.ok === true) {
                        session.checkedCoverageReviewWindowIds.add(validation.value.review_window_id)
                    }
                    return output
                })

                return jsonOutput(output)
            }), {
                name: 'check_translation_rule_coverage',
                description: 'Review-only read tool that checks whether current-window Terms have translation_rule coverage; proper-name gaps may block, newer item/repeated/system checks are warning-only.',
                schema: looseCheckTranslationRuleCoverageSchema,
            }),
            tool(async input => withToolLogging('append_review_entry', input, onToolEvent, async () => {
                const validation = validateAppendReviewEntryInput(input)

                if (!validation.ok) {
                    return jsonOutput(validationErrorOutput(validation.errors))
                }

                const output = await updateGlossaryState(root, state => runAppendReviewEntry(projectRoot, manualTransFile, evidenceScope, state, session, validation.value))
                return jsonOutput(output)
            }), {
                name: 'append_review_entry',
                description: 'Review-only tool to append an Entry to an active Term through the current review window. Evidence is required except for transliteration/project_convention translation_rule naming decisions. The Entry is applied atomically when review commits.',
                schema: looseAppendReviewEntrySchema,
            }),
        ] : []),
        tool(async input => withToolLogging('query_term_merge_proposals', input, onToolEvent, async () => {
            const validation = validateQueryTermMergeProposalsInput(input)

            if (!validation.ok) {
                return jsonOutput({
                    ok: false,
                    code: 'invalid_query',
                    errors: validation.errors,
                })
            }

            const output = await readGlossaryState(root, state => {
                const result = runQueryTermMergeProposals(state, validation.value)

                if (validation.value.source_term_id && validation.value.target_term_id) {
                    session.queriedMergeProposalPairs.add(termPairKey(validation.value.source_term_id, validation.value.target_term_id))
                }

                return result
            })

            return jsonOutput(output)
        }), {
            name: 'query_term_merge_proposals',
            description: 'Query candidate/reviewed Term merge proposals by term ids and status. Call this before creating a merge proposal for a term pair.',
            schema: looseQueryTermMergeProposalsSchema,
        }),
        tool(async input => withToolLogging('create_or_get_term', input, onToolEvent, async () => {
            const validation = validateCreateOrGetTermInput(input)

            if (!validation.ok) {
                return jsonOutput(validationOutput(validation.errors))
            }

            const value = withCreatedFromBatchId(validation.value, options.createdFromBatchId)
            const output = await updateGlossaryState(root, state => runMutableOrCache(state, options.reviewMode, 'create_or_get_term', value as unknown as Record<string, unknown>, () => runCreateOrGetTerm(state, session, value, options.enableCharacterGenderWarnings)))
            return jsonOutput(output)
        }), {
            name: 'create_or_get_term',
            description: 'Idempotently create or get an active glossary Term after a prior query_glossary_terms call for the candidate.',
            schema: looseCreateOrGetTermSchema,
        }),
        ...(options.enableBatchTools ? [
            tool(async input => withToolLogging('create_or_get_terms', input, onToolEvent, async () => {
                const validation = validateCreateOrGetTermsInput(input)

                if (!validation.ok) {
                    return jsonOutput(validationOutput(validation.errors))
                }

                const value: CreateOrGetTermsInput = {
                    items: validation.value.items.map(item => withCreatedFromBatchId(item, options.createdFromBatchId)),
                }
                const output = await updateGlossaryState(root, state => runMutableOrCache(state, options.reviewMode, 'create_or_get_terms', value as unknown as Record<string, unknown>, () => runCreateOrGetTerms(state, session, value)))
                return jsonOutput(output)
            }), {
                name: 'create_or_get_terms',
                description: `Batch create-or-get active glossary Terms after prior query_glossary_terms calls. Each item is applied independently; max ${MAX_BATCH_TOOL_ITEMS} items.`,
                schema: looseCreateOrGetTermsSchema,
            }),
        ] : []),
        tool(async input => withToolLogging('update_term_metadata', input, onToolEvent, async () => {
            const validation = validateUpdateTermMetadataInput(input)

            if (!validation.ok) {
                return jsonOutput(validationOutput(validation.errors))
            }

            const value = withUpdatedFromBatchId(validation.value, options.createdFromBatchId)
            const output = await updateGlossaryState(root, state => runMutableOrCache(state, options.reviewMode, 'update_term_metadata', compactUpdateTermMetadataInputForCache(value), () => runUpdateTermMetadata(state, session, value, options.enableCharacterGenderWarnings)))
            return jsonOutput(output)
        }), {
            name: 'update_term_metadata',
            description: 'Update Term metadata such as aliases, term type, deprecation, rejection, or merge target after querying the Term.',
            schema: looseUpdateTermMetadataSchema,
        }),
        tool(async input => withToolLogging('update_term_entries', input, onToolEvent, async () => {
            const validation = validateUpdateTermEntriesInput(input)

            if (!validation.ok) {
                return jsonOutput(validationErrorOutput(validation.errors))
            }

            const value = withCreatedFromBatchId(validation.value, options.createdFromBatchId)
            const output = await updateGlossaryState(root, state => runMutableOrCache(state, options.reviewMode, 'update_term_entries', value as unknown as Record<string, unknown>, () => runUpdateTermEntries(projectRoot, manualTransFile, evidenceScope, state, session, value, options.enableTargetTermRevisionCheck)))
            return jsonOutput(output)
        }), {
            name: 'update_term_entries',
            description: 'Append one candidate Entry and Evidence for a queried Term. Append-only; entry type fields are strictly validated.',
            schema: looseUpdateTermEntriesSchema,
        }),
        ...(options.enableBatchTools ? [
            tool(async input => withToolLogging('append_term_entries_batch', input, onToolEvent, async () => {
                const validation = validateAppendTermEntriesBatchInput(input)

                if (!validation.ok) {
                    return jsonOutput(validationErrorOutput(validation.errors))
                }

                const value: AppendTermEntriesBatchInput = {
                    items: validation.value.items.map(item => withCreatedFromBatchId(item, options.createdFromBatchId)),
                }
                const output = await updateGlossaryState(root, state => runMutableOrCache(state, options.reviewMode, 'append_term_entries_batch', value as unknown as Record<string, unknown>, () => runAppendTermEntriesBatch(projectRoot, manualTransFile, evidenceScope, state, session, value, options.enableTargetTermRevisionCheck)))
                return jsonOutput(output)
            }), {
                name: 'append_term_entries_batch',
                description: `Batch append candidate Entries and Evidence for queried Terms. Each item is applied independently; max ${MAX_BATCH_TOOL_ITEMS} items.`,
                schema: looseAppendTermEntriesBatchSchema,
            }),
        ] : []),
        tool(async input => withToolLogging('create_term_merge_proposal', input, onToolEvent, async () => {
            const validation = validateCreateTermMergeProposalInput(input)

            if (!validation.ok) {
                return jsonOutput(validationErrorOutput(validation.errors))
            }

            const value = withCreatedFromBatchId(validation.value, options.createdFromBatchId)
            const output = await updateGlossaryState(root, state => runMutableOrCache(state, options.reviewMode, 'create_term_merge_proposal', value as unknown as Record<string, unknown>, () => runCreateTermMergeProposal(projectRoot, manualTransFile, evidenceScope, state, session, value)))
            return jsonOutput(output)
        }), {
            name: 'create_term_merge_proposal',
            description: 'Create an evidence-backed candidate proposal to merge two existing Terms. This does not apply the merge; review handles approval and application.',
            schema: looseCreateTermMergeProposalSchema,
        }),
        ...(options.reviewMode ? [
            tool(async input => withToolLogging('review_entries_batch', input, onToolEvent, async () => {
                const validation = validateReviewEntriesBatchInput(input)

                if (!validation.ok) {
                    return jsonOutput(validationErrorOutput(validation.errors))
                }

                const output = await updateGlossaryState(root, state => runReviewEntriesBatch(state, validation.value))
                return jsonOutput(output)
            }), {
                name: 'review_entries_batch',
                description: `Register review decisions for Entries in the active review window. Actions are applied atomically by the program after review-agent finishes. Max ${MAX_BATCH_TOOL_ITEMS} actions.`,
                schema: looseReviewEntriesBatchSchema,
            }),
            tool(async input => withToolLogging('review_terms_batch', input, onToolEvent, async () => {
                const validation = validateReviewTermsBatchInput(input)

                if (!validation.ok) {
                    return jsonOutput(validationErrorOutput(validation.errors))
                }

                const output = await updateGlossaryState(root, state => runReviewTermsBatch(state, validation.value, options.enableCharacterGenderWarnings))
                return jsonOutput(output)
            }), {
                name: 'review_terms_batch',
                description: `Register review decisions for Terms in the active review window. Actions are applied atomically by the program after review-agent finishes. Max ${MAX_BATCH_TOOL_ITEMS} actions.`,
                schema: looseReviewTermsBatchSchema,
            }),
            tool(async input => withToolLogging('submit_glossary_review', input, onToolEvent, async () => {
                const validation = await validateSubmitGlossaryReviewInput(root, session, input, {
                    summaryMaxChars: options.summaryMaxChars ?? null,
                    deferredNotesMaxItems: options.deferredNotesMaxItems ?? null,
                })

                if (!validation.ok) {
                    return jsonOutput(validationErrorOutput(validation.errors))
                }

                if (reviewSubmitted) {
                    return jsonOutput({
                        ok: false,
                        code: 'already_submitted',
                        message: 'submit_glossary_review has already been accepted for this review-agent session.',
                        retry_required: false,
                    })
                }

                reviewSubmitted = true
                options.onReviewSubmitted?.(validation.value)
                const characterContextWarnings = options.enableCharacterGenderWarnings
                    ? await readGlossaryState(root, state => {
                        const window = state.review.active_window?.review_window_id === validation.value.review_window_id
                            ? state.review.active_window
                            : state.review.windows.find(item => item.review_window_id === validation.value.review_window_id) ?? null

                        if (!window) {
                            return []
                        }

                        return collectCharacterContextWarningsForWindow(state, window)
                    })
                    : []

                return jsonOutput({
                    ok: true,
                    ...(characterContextWarnings.length > 0 ? { character_context_warnings: characterContextWarnings } : {}),
                    retry_required: false,
                })
            }), {
                name: 'submit_glossary_review',
                description: appendSubmitLimitDescription(
                    'Submit the active glossary review as complete. This is the only completion signal; summary is used for the review commit log.',
                    options,
                ),
                schema: looseSubmitGlossaryReviewSchema,
            }),
        ] : []),
        ...(!options.reviewMode && !options.readOnly && options.workerBatchId ? [
            tool(async input => withToolLogging('submit_glossary_worker_batch', input, onToolEvent, async () => {
                const validation = validateSubmitGlossaryWorkerBatchInput(input, options)

                if (!validation.ok) {
                    return jsonOutput(validationErrorOutput(validation.errors))
                }

                if (workerBatchSubmitted) {
                    return jsonOutput({
                        ok: false,
                        code: 'already_submitted',
                        message: 'submit_glossary_worker_batch has already been accepted for this worker-agent session.',
                        retry_required: false,
                    })
                }

                workerBatchSubmitted = true
                options.onWorkerBatchSubmitted?.(validation.value)

                return jsonOutput({
                    ok: true,
                    retry_required: false,
                })
            }), {
                name: 'submit_glossary_worker_batch',
                description: appendSubmitLimitDescription(
                    'Submit the current glossary worker batch as complete. This is the only completion signal; it does not create Terms, Entries, or Evidence.',
                    options,
                ),
                schema: looseSubmitGlossaryWorkerBatchSchema,
            }),
        ] : []),
        tool(async input => withToolLogging('submit_glossary_plan', input, onToolEvent, async () => {
            const validation = validateGlossaryPlanInput(input)

            if (!validation.ok) {
                return jsonOutput({
                    ok: false,
                    errors: validation.errors,
                    retry_required: true,
                } satisfies SubmitGlossaryPlanOutput)
            }

            const plan = await appendGlossaryPlan(root, validation.value)
            planSubmitted?.(plan)

            return jsonOutput({
                ok: true,
                plan_id: plan.plan_id,
                retry_required: false,
            } satisfies SubmitGlossaryPlanOutput)
        }), {
            name: 'submit_glossary_plan',
            description: 'Submit and persist the glossary preflight extraction plan. This does not create Term, Entry, or Evidence records.',
            schema: looseGlossaryPlanSchema,
        }),
    ]

    if (options.readOnly || options.reviewMode) {
        const disallowedTools = new Set([
            'create_or_get_term',
            'create_or_get_terms',
            'update_term_metadata',
            'update_term_entries',
            'append_term_entries_batch',
            'create_term_merge_proposal',
            'submit_glossary_plan',
        ])

        if (options.readOnly) {
            disallowedTools.add('list_review_candidates')
            disallowedTools.add('review_entries_batch')
            disallowedTools.add('review_terms_batch')
            disallowedTools.add('append_review_entry')
            disallowedTools.add('submit_glossary_review')
            disallowedTools.add('submit_glossary_worker_batch')
        }

        return tools.filter(item => !disallowedTools.has(item.name))
    }

    return tools
}

function normalizeGlossaryToolsOptions (
    root: string,
    manualTransFileOrPlanSubmitted: string|CreateGlossaryToolsOptions|undefined,
    onPlanSubmitted?: (plan: GlossaryPlan) => void,
): Required<Pick<CreateGlossaryToolsOptions, 'manualTransFile'|'projectRoot'|'enableBatchTools'|'enableTargetTermRevisionCheck'|'enableCharacterGenderWarnings'|'readOnly'>> & Omit<CreateGlossaryToolsOptions, 'manualTransFile'|'projectRoot'|'enableBatchTools'|'enableTargetTermRevisionCheck'|'enableCharacterGenderWarnings'|'readOnly'> {
    if (typeof manualTransFileOrPlanSubmitted === 'string') {
        return {
            manualTransFile: manualTransFileOrPlanSubmitted,
            projectRoot: root,
            onPlanSubmitted,
            enableBatchTools: false,
            enableTargetTermRevisionCheck: true,
            enableCharacterGenderWarnings: false,
            reviewMode: false,
            readOnly: false,
            summaryMaxChars: null,
            deferredNotesMaxItems: null,
        }
    }

    if (!manualTransFileOrPlanSubmitted?.manualTransFile) {
        throw new Error('createGlossaryTools requires manualTransFile.')
    }

    return {
        manualTransFile: manualTransFileOrPlanSubmitted.manualTransFile,
        projectRoot: manualTransFileOrPlanSubmitted?.projectRoot ?? root,
        onPlanSubmitted: manualTransFileOrPlanSubmitted?.onPlanSubmitted ?? onPlanSubmitted,
        evidenceScope: manualTransFileOrPlanSubmitted?.evidenceScope,
        enableBatchTools: manualTransFileOrPlanSubmitted?.enableBatchTools ?? false,
        enableTargetTermRevisionCheck: manualTransFileOrPlanSubmitted?.enableTargetTermRevisionCheck ?? true,
        enableCharacterGenderWarnings: manualTransFileOrPlanSubmitted?.enableCharacterGenderWarnings ?? false,
        reviewMode: manualTransFileOrPlanSubmitted?.reviewMode ?? false,
        readOnly: manualTransFileOrPlanSubmitted?.readOnly ?? false,
        createdFromBatchId: manualTransFileOrPlanSubmitted?.createdFromBatchId,
        onWorkerBatchSubmitted: manualTransFileOrPlanSubmitted?.onWorkerBatchSubmitted,
        onReviewSubmitted: manualTransFileOrPlanSubmitted?.onReviewSubmitted,
        workerBatchId: manualTransFileOrPlanSubmitted?.workerBatchId,
        workerBatchNumber: manualTransFileOrPlanSubmitted?.workerBatchNumber,
        summaryMaxChars: manualTransFileOrPlanSubmitted?.summaryMaxChars ?? null,
        deferredNotesMaxItems: manualTransFileOrPlanSubmitted?.deferredNotesMaxItems ?? null,
        requireWorkerBatchRead: manualTransFileOrPlanSubmitted?.requireWorkerBatchRead,
    }
}

function appendSubmitLimitDescription (description: string, options: CreateGlossaryToolsOptions): string {
    const limits: string[] = []

    if (options.summaryMaxChars !== null && options.summaryMaxChars !== undefined) {
        limits.push(`summary and each deferred_notes item must be at most ${options.summaryMaxChars} Unicode code units`)
    }

    if (options.deferredNotesMaxItems !== null && options.deferredNotesMaxItems !== undefined) {
        limits.push(`deferred_notes must contain at most ${options.deferredNotesMaxItems} items`)
    }

    return limits.length === 0
        ? description
        : `${description} Limits: ${limits.join('; ')}.`
}

function withCreatedFromBatchId<T extends { created_from: Record<string, unknown> }> (input: T, batchId: string|undefined): T {
    if (!batchId) {
        return input
    }

    return {
        ...input,
        created_from: {
            ...input.created_from,
            batch_id: batchId,
        },
    }
}

function withUpdatedFromBatchId<T extends { updated_from: Record<string, unknown> }> (input: T, batchId: string|undefined): T {
    if (!batchId) {
        return input
    }

    return {
        ...input,
        updated_from: {
            ...input.updated_from,
            batch_id: batchId,
        },
    }
}

export async function markGlossaryWorkerBatchCompleted (root: string, batchNumber: number): Promise<void> {
    await updateGlossaryState(root, state => {
        if (!state.review.completed_batches.includes(batchNumber)) {
            state.review.completed_batches.push(batchNumber)
            state.review.completed_batches.sort((left, right) => left - right)
            state.meta.updated_at = new Date().toISOString()
        }
    })
}

export async function getGlossaryPendingReviewBatchCount (root: string, batchNumbers?: number[]): Promise<number> {
    return readGlossaryState(root, state => getPendingReviewBatchNumbers(state, batchNumbers).length)
}

export async function beginGlossaryReviewWindow (root: string, batchNumbers?: number[]): Promise<GlossaryReviewWindow|null> {
    return updateGlossaryState(root, state => {
        if (state.review.active_window) {
            return structuredClone(state.review.active_window) as GlossaryReviewWindow
        }

        const completedBatchNumbers = getPendingReviewBatchNumbers(state, batchNumbers)

        if (completedBatchNumbers.length === 0) {
            return null
        }

        const now = new Date().toISOString()
        const window: GlossaryReviewWindow = {
            review_window_id: createNextReviewWindowId(state),
            status: 'running',
            started_at: now,
            last_reviewed_batch_number: state.review.last_reviewed_batch_number,
            completed_batch_numbers: completedBatchNumbers,
            pending_entry_actions: [],
            pending_term_actions: [],
        }

        // No worker/review parallelism today; keep freeze cache for future use.
        state.review.frozen = false
        state.review.active_window = window
        state.review.windows.push(window)
        state.meta.updated_at = now

        return structuredClone(window) as GlossaryReviewWindow
    })
}

export async function commitGlossaryReviewWindow (root: string, reviewWindowId: string, summary: string, manualTransFile: string, enableCharacterGenderWarnings = false): Promise<GlossaryReviewCommitResult> {
    return updateGlossaryState(root, async state => {
        const window = getActiveReviewWindow(state, reviewWindowId)
        const entryActionCount = window.pending_entry_actions.length
        const termActionCount = window.pending_term_actions.length

        applyReviewEntryActions(state, window.pending_entry_actions)
        applyReviewTermActions(state, window.pending_term_actions)

        // Cache replay is inactive until worker/review parallelism returns.
        const cacheStats = { applied: 0, conflicted: 0 }
        const warnings = collectActiveTermWithoutEffectiveEntryWarnings(state)
        const characterContextWarnings = enableCharacterGenderWarnings
            ? collectCharacterContextWarnings({
                terms: state.terms,
                entries: state.entries,
                termIds: collectReviewWindowRelatedTermIds(state, window),
            })
            : []
        const now = new Date().toISOString()
        const maxReviewedBatch = Math.max(window.last_reviewed_batch_number, ...window.completed_batch_numbers)

        window.status = 'completed'
        window.completed_at = now
        window.summary = summary
        state.review.last_reviewed_batch_number = maxReviewedBatch
        state.review.frozen = false
        state.review.active_window = null
        state.review.logs.push({
            review_window_id: window.review_window_id,
            status: 'completed',
            started_at: window.started_at,
            completed_at: now,
            summary,
            entry_action_count: entryActionCount,
            term_action_count: termActionCount,
            cache_applied_count: cacheStats.applied,
            cache_conflict_count: cacheStats.conflicted,
        })
        state.meta.updated_at = now

        return {
            review_window_id: window.review_window_id,
            entry_action_count: entryActionCount,
            term_action_count: termActionCount,
            cache_applied_count: cacheStats.applied,
            cache_conflict_count: cacheStats.conflicted,
            ...(warnings.length > 0 ? { warnings } : {}),
            ...(characterContextWarnings.length > 0 ? { character_context_warnings: characterContextWarnings } : {}),
        }
    })
}

export async function failGlossaryReviewWindow (root: string, reviewWindowId: string, errorMessage: string): Promise<void> {
    await updateGlossaryState(root, state => {
        const window = getActiveReviewWindow(state, reviewWindowId)
        const now = new Date().toISOString()

        window.status = 'failed'
        window.failed_at = now
        window.error = errorMessage
        state.review.frozen = false
        state.review.active_window = null
        state.review.logs.push({
            review_window_id: window.review_window_id,
            status: 'failed',
            started_at: window.started_at,
            completed_at: now,
            error: errorMessage,
            entry_action_count: window.pending_entry_actions.length,
            term_action_count: window.pending_term_actions.length,
            cache_applied_count: 0,
            cache_conflict_count: 0,
        })
        state.meta.updated_at = now
    })
}

async function runQueryGlossaryTerms (state: GlossaryState, input: ValidatedQueryGlossaryTermsInput): Promise<QueryGlossaryTermsOutput> {
    const matchedTerms: QueryGlossaryTermsOutput['matched_terms'] = []
    const seenTermIds = new Set<string>()
    const termStatuses = input.term_statuses ? new Set<TermStatus>(input.term_statuses) : null
    const yieldController = createYieldController()

    for (const query of input.queries) {
        for (const mode of query.match_modes) {
            const matcher = createTermMatcher(query.text, mode, query.case_sensitive)

            for (const [termIndex, term] of state.terms.entries()) {
                await yieldController.maybeYield(termIndex)
                if (seenTermIds.has(term.term_id) || term.source_language !== input.source_language) {
                    continue
                }

                if (termStatuses && !termStatuses.has(term.status)) {
                    continue
                }

                const matchedText = matcher(term)

                if (!matchedText) {
                    continue
                }

                const hideRejectedDetails = term.status === 'rejected' && !input.include_rejected_details
                const entries = input.include_entries && !hideRejectedDetails
                    ? state.entries.filter(entry => entry.term_id === term.term_id && entryStatusAllowed(entry, input.entry_statuses))
                    : []
                const evidence = input.include_evidence && !hideRejectedDetails
                    ? state.evidence.filter(item => item.term_id === term.term_id && evidenceEntryAllowed(item, entries, input.include_entries, input.entry_statuses, state))
                    : []

                matchedTerms.push({
                    term,
                    match: {
                        query_text: query.text,
                        match_mode: mode,
                        matched_text: matchedText,
                    },
                    entries,
                    evidence,
                })
                seenTermIds.add(term.term_id)

                if (matchedTerms.length >= input.limit) {
                    return {
                        ok: true,
                        matched_terms: matchedTerms,
                        warnings: [],
                    }
                }
            }
        }
    }

    return {
        ok: true,
        matched_terms: matchedTerms,
        warnings: [],
    }
}

async function runSearchGlossaryEntries (state: GlossaryState, input: SearchGlossaryEntriesInput): Promise<{
    ok: true
    include_term: boolean
    matched_entries: {
        entry: GlossaryEntry
        term: GlossaryTerm
        evidence: GlossaryEvidence[]
    }[]
    warnings: string[]
}> {
    const matchedEntries: {
        entry: GlossaryEntry
        term: GlossaryTerm
        evidence: GlossaryEvidence[]
    }[] = []
    const yieldController = createYieldController()

    for (const [entryIndex, entry] of state.entries.entries()) {
        await yieldController.maybeYield(entryIndex)
        const term = state.terms.find(candidate => candidate.term_id === entry.term_id)

        if (!entryMatchesFilter(entry, input.entry_filter) || !termMatchesFilter(term, input.term_filter)) {
            continue
        }

        if (!term) {
            continue
        }

        matchedEntries.push({
            entry,
            term,
            evidence: input.include_evidence
                ? state.evidence.filter(evidence => evidence.entry_id === entry.entry_id)
                : [],
        })

        if (matchedEntries.length >= input.limit) {
            break
        }
    }

    return {
        ok: true,
        include_term: input.include_term,
        matched_entries: matchedEntries,
        warnings: [],
    }
}

async function formatQueryGlossaryTermsOutput (root: string, manualTransFile: string, output: QueryGlossaryTermsOutput): Promise<unknown> {
    const evidenceFormatter = await createEvidenceFormatter(root, manualTransFile)
    return {
        ...output,
        matched_terms: output.matched_terms.map(match => ({
            term: formatTermForAgent(match.term),
            match: match.match,
            entries: match.entries.map(entry => formatEntryForAgent(entry, match.term)),
            evidence: match.evidence.map(evidenceFormatter),
        })),
    }
}

async function formatSearchGlossaryEntriesOutput (root: string, manualTransFile: string, output: Awaited<ReturnType<typeof runSearchGlossaryEntries>>): Promise<unknown> {
    const evidenceFormatter = await createEvidenceFormatter(root, manualTransFile)
    return {
        ok: output.ok,
        matched_entries: output.matched_entries.map(match => ({
            entry: formatEntryForAgent(match.entry, match.term),
            ...(output.include_term ? { term: formatTermForAgent(match.term) } : {}),
            evidence: match.evidence.map(evidenceFormatter),
        })),
        warnings: output.warnings,
    }
}

async function formatReviewCandidatesOutput (root: string, manualTransFile: string, output: unknown): Promise<unknown> {
    if (!isRecord(output) || output.ok !== true) {
        return output
    }

    const evidenceFormatter = await createEvidenceFormatter(root, manualTransFile)
    const sourceRefFormatter = await createSourceRefFormatter(root, manualTransFile)
    return {
        ...output,
        evidence: Array.isArray(output.evidence) ? output.evidence.map(item => isGlossaryEvidenceLike(item) ? evidenceFormatter(item) : item) : output.evidence,
        merge_proposals: Array.isArray(output.merge_proposals)
            ? output.merge_proposals.map(proposal => isRecord(proposal) && Array.isArray(proposal.evidence)
                ? { ...proposal, evidence: proposal.evidence.map(item => isRecord(item) && isRecord(item.source_ref) ? { ...item, source_ref: sourceRefFormatter(item.source_ref) } : item) }
                : proposal)
            : output.merge_proposals,
        pending_entry_actions: Array.isArray(output.pending_entry_actions)
            ? output.pending_entry_actions.map(action => isRecord(action) && Array.isArray(action.evidence)
                ? { ...action, evidence: action.evidence.map(item => isRecord(item) && isRecord(item.source_ref) ? { ...item, source_ref: sourceRefFormatter(item.source_ref) } : item) }
                : action)
            : output.pending_entry_actions,
    }
}

function formatOptionalTermForAgent (term: GlossaryTerm|undefined): unknown {
    return term ? formatTermForAgent(term) : null
}

function formatEntryForAgentByState (state: GlossaryState, entry: GlossaryEntry): unknown {
    const term = state.terms.find(item => item.term_id === entry.term_id)
    return term ? formatEntryForAgent(entry, term) : entry
}

async function createEvidenceFormatter (root: string, manualTransFile: string): Promise<(evidence: GlossaryEvidence) => unknown> {
    const sourceRefFormatter = await createSourceRefFormatter(root, manualTransFile)

    return evidence => ({
        ...evidence,
        source_ref: sourceRefFormatter(evidence.source_ref),
    })
}

async function createSourceRefFormatter (root: string, manualTransFile: string): Promise<(sourceRef: { source_file_id?: string, file_id?: string, key_index?: number, key_hash?: string, span?: { start: number, end: number } }) => Record<string, unknown>> {
    const manualTrans = await loadManualTransData(root, manualTransFile).catch(() => null)
    const filteredIndexByRawIndex = manualTrans ? await createFilteredIndexByRawIndex(root, manualTrans) : new Map<number, number>()

    return sourceRef => {
        const rawIndex = sourceRef.key_index
        const filteredKeyIndex = typeof rawIndex === 'number' ? filteredIndexByRawIndex.get(rawIndex) : undefined
        const output: Record<string, unknown> = {
            source_file_id: sourceRef.source_file_id ?? sourceRef.file_id,
            key_hash: sourceRef.key_hash,
            span: sourceRef.span,
        }

        if (filteredKeyIndex !== undefined) {
            output.filtered_key_index = filteredKeyIndex
        } else {
            output.filtered_key_index_unavailable = true
        }

        return output
    }
}

function isGlossaryEvidenceLike (value: unknown): value is GlossaryEvidence {
    return isRecord(value) && isRecord(value.source_ref) && typeof value.quote === 'string' && typeof value.reason === 'string'
}

function runQueryTermMergeProposals (state: GlossaryState, input: QueryTermMergeProposalsInput): {
    ok: true
    matched_proposals: {
        proposal: GlossaryMergeProposal
        source_term?: unknown
        target_term?: unknown
    }[]
    warnings: string[]
} {
    const statuses = input.statuses ? new Set(input.statuses) : null
    const matchedProposals = state.merge_proposals
        .filter(proposal => mergeProposalMatchesFilter(proposal, input, statuses))
        .slice(0, input.limit)
        .map(proposal => ({
            proposal,
            ...(input.include_terms ? {
                source_term: formatOptionalTermForAgent(state.terms.find(term => term.term_id === proposal.source_term_id)),
                target_term: formatOptionalTermForAgent(state.terms.find(term => term.term_id === proposal.target_term_id)),
            } : {}),
        }))

    return {
        ok: true,
        matched_proposals: matchedProposals,
        warnings: [],
    }
}

async function runGetEvidenceContext (
    root: string,
    manualTransFile: string,
    state: GlossaryState,
    session: GlossaryToolSession,
    input: EvidenceContextInput,
): Promise<{
    ok: true
    contexts: unknown[]
    warnings: string[]
}|{
    ok: false
    code: 'invalid_evidence_id'|'window_too_large'|'source_not_found'
    errors: ValidationError[]
}> {
    const missingEvidenceIds = input.evidence_ids.filter(id => !session.returnedEvidenceIds.has(id))

    if (missingEvidenceIds.length > 0) {
        return {
            ok: false,
            code: 'invalid_evidence_id',
            errors: missingEvidenceIds.map(id => ({
                field: 'evidence_ids',
                message: `${id} 不是当前 agent 通过 query/search 获得的 Evidence id。`,
            })),
        }
    }

    let manualTrans: ManualTransData

    try {
        manualTrans = await loadManualTransData(root, manualTransFile)
    } catch (error) {
        return {
            ok: false,
            code: 'source_not_found',
            errors: [
                {
                    field: 'source_ref.source_file_id',
                    message: error instanceof Error ? error.message : String(error),
                },
            ],
        }
    }
    const filteredIndexByRawIndex = await createFilteredIndexByRawIndex(root, manualTrans)

    const contexts: unknown[] = []

    for (const evidenceId of input.evidence_ids) {
        const evidence = state.evidence.find(item => item.evidence_id === evidenceId)

        if (!evidence) {
            return {
                ok: false,
                code: 'invalid_evidence_id',
                errors: [
                    {
                        field: 'evidence_ids',
                        message: `${evidenceId} 不存在。`,
                    },
                ],
            }
        }

        const sourceFileId = getEvidenceSourceFileId(evidence)

        if (!sourceFileId) {
            return {
                ok: false,
                code: 'source_not_found',
                errors: [
                    {
                        field: 'source_ref.source_file_id',
                        message: `${evidenceId} 缺少可解析的 source_ref.source_file_id。`,
                    },
                ],
            }
        }

        if (!evidence.source_ref?.key_hash && sourceFileId && normalizeProjectFileId(sourceFileId) !== normalizeProjectFileId(manualTrans.relativePath)) {
            return {
                ok: false,
                code: 'source_not_found',
                errors: [
                    {
                        field: 'source_ref.source_file_id',
                        message: `${sourceFileId} 与当前配置源文件 ${manualTrans.relativePath} 不一致。`,
                    },
                ],
            }
        }

        const keyIndex = evidence.source_ref?.key_index

        if (typeof keyIndex !== 'number' || keyIndex < 0 || keyIndex >= manualTrans.keys.length) {
            return {
                ok: false,
                code: 'source_not_found',
                errors: [
                    {
                        field: 'source_ref.filtered_key_index',
                        message: `${evidenceId} 无法定位到源 key。`,
                    },
                ],
            }
        }

        const keyText = manualTrans.keys[keyIndex]

        if (evidence.source_ref?.key_hash && normalizeKeyHash(evidence.source_ref.key_hash) !== sha256Text(keyText)) {
            return {
                ok: false,
                code: 'source_not_found',
                errors: [
                    {
                        field: 'source_ref.key_hash',
                        message: `${evidenceId} 的 key_hash 与当前源 key 不一致。`,
                    },
                ],
            }
        }

        const filteredKeyIndex = filteredIndexByRawIndex.get(keyIndex)

        if (filteredKeyIndex === undefined) {
            return {
                ok: false,
                code: 'source_not_found',
                errors: [
                    {
                        field: 'source_ref.filtered_key_index',
                        message: `${evidenceId} 无法映射到过滤后的 key index。`,
                    },
                ],
            }
        }

        const filteredEntries = Array.from(filteredIndexByRawIndex.entries())
            .map(([rawIndex, filteredIndex]) => ({ rawIndex, filteredIndex }))
            .sort((left, right) => left.filteredIndex - right.filteredIndex)
        const startFilteredIndex = Math.max(0, filteredKeyIndex - input.window_before)
        const endFilteredIndex = filteredKeyIndex + input.window_after
        const contextRawIndexes = filteredEntries
            .filter(item => item.filteredIndex >= startFilteredIndex && item.filteredIndex <= endFilteredIndex)
            .map(item => item.rawIndex)
        const contextItems = createContextKeyItemsByRawIndexes(manualTrans, contextRawIndexes, input.max_chars_per_key, input.include_values, filteredIndexByRawIndex)
        const anchorOffset = contextRawIndexes.indexOf(keyIndex)
        const startIndex = Math.min(...contextRawIndexes)
        const endIndex = Math.max(...contextRawIndexes)
        session.allowedEvidenceRanges.push({
            source_file_id: manualTrans.relativePath,
            start_index: startIndex,
            end_index: endIndex,
            allowed_key_indices: contextRawIndexes,
        })

        contexts.push({
            evidence_id: evidenceId,
            anchor: {
                source_file_id: manualTrans.relativePath,
                filtered_key_index: filteredKeyIndex,
                key_hash: sha256SourceText(manualTrans.keys[keyIndex]),
            },
            before: contextItems.slice(0, anchorOffset),
            anchor_key: contextItems[anchorOffset],
            after: contextItems.slice(anchorOffset + 1),
            truncated: contextItems.some(item => isRecord(item) && item.truncated === true),
        })
    }

    return {
        ok: true,
        contexts,
        warnings: [],
    }
}

async function runListReviewCandidates (
    root: string,
    manualTransFile: string,
    state: GlossaryState,
    input: ListReviewCandidatesInput,
    enableCharacterGenderWarnings = false,
): Promise<unknown> {
    const window = input.review_window_id
        ? state.review.active_window?.review_window_id === input.review_window_id
            ? state.review.active_window
            : state.review.windows.find(item => item.review_window_id === input.review_window_id)
        : state.review.active_window

    if (!window) {
        return {
            ok: false,
            code: 'no_active_review_window',
            message: '当前没有可用的 review window。',
            retry_required: false,
        }
    }

    const batchNumbers = new Set(window.completed_batch_numbers)
    const batchIds = new Set(window.completed_batch_numbers.map(formatBatchId))
    const newEntries = state.entries
        .filter(entry => batchIds.has(readCreatedBatchId(entry.created_from)))
        .slice(0, input.limit)
    const newTermIds = new Set([
        ...state.terms.filter(term => batchIds.has(readCreatedBatchId(term.created_from))).map(term => term.term_id),
        ...state.terms.filter(term => hasTermUpdatedFromAnyBatch(term, batchIds)).map(term => term.term_id),
        ...newEntries.map(entry => entry.term_id),
    ])
    const newTerms = state.terms
        .filter(term => newTermIds.has(term.term_id))
        .slice(0, input.limit)
    const newEntryIds = new Set(newEntries.map(entry => entry.entry_id))
    const newEvidence = state.evidence
        .filter(evidence => newEntryIds.has(evidence.entry_id))
        .slice(0, input.limit)
    const structuralIssues = await collectReviewStructuralIssues(root, manualTransFile, state, window, newTerms, newEntries, newEvidence)
    const characterContextWarnings = enableCharacterGenderWarnings
        ? collectCharacterContextWarnings({
            terms: state.terms.map(term => applyPendingTermActionsForCharacterContext(term, window.pending_term_actions)),
            entries: state.entries,
            termIds: newTermIds,
        })
        : []

    return {
        ok: true,
        review_window_id: window.review_window_id,
        frozen: state.review.frozen,
        last_reviewed_batch_number: window.last_reviewed_batch_number,
        completed_batch_numbers: Array.from(batchNumbers).sort((left, right) => left - right),
        unreviewed_batch_numbers: getPendingReviewBatchNumbers(state),
        counts: {
            terms: newTerms.length,
            entries: newEntries.length,
            evidence: newEvidence.length,
            empty_terms: newTerms.filter(term => term.entry_ids.length === 0).length,
            structural_issues: structuralIssues.length,
            conflicts: state.review.conflicts.length,
            cached_worker_writes: state.review.cached_worker_writes.filter(write => write.status === 'pending').length,
            merge_proposals: state.merge_proposals.filter(proposal => proposal.status === 'candidate').length,
        },
        terms: input.include_terms ? newTerms.map(formatTermForAgent) : [],
        entries: input.include_entries ? newEntries.map(entry => formatEntryForAgentByState(state, entry)) : [],
        evidence: input.include_evidence ? newEvidence : [],
        empty_terms: input.include_terms ? newTerms.filter(term => term.entry_ids.length === 0).map(formatTermForAgent) : [],
        structural_issues: structuralIssues,
        conflicts: state.review.conflicts.slice(-input.limit),
        merge_proposals: state.merge_proposals.filter(proposal => proposal.status === 'candidate').slice(0, input.limit),
        pending_entry_actions: window.pending_entry_actions,
        pending_term_actions: window.pending_term_actions,
        ...(characterContextWarnings.length > 0 ? { character_context_warnings: characterContextWarnings } : {}),
        warnings: [],
    }
}

function recordReviewCandidateEvidence (session: GlossaryToolSession, output: unknown): void {
    if (!isRecord(output) || output.ok !== true) {
        return
    }

    const evidence = output.evidence
    if (Array.isArray(evidence)) {
        for (const item of evidence) {
            if (isRecord(item) && typeof item.evidence_id === 'string') {
                session.returnedEvidenceIds.add(item.evidence_id)
            }
        }
    }

    const entries = output.entries
    if (Array.isArray(entries)) {
        for (const entry of entries) {
            if (!isRecord(entry) || !Array.isArray(entry.evidence_ids)) {
                continue
            }

            for (const evidenceId of entry.evidence_ids) {
                if (typeof evidenceId === 'string') {
                    session.returnedEvidenceIds.add(evidenceId)
                }
            }
        }
    }
}

function runCreateOrGetTerm (
    state: GlossaryState,
    session: GlossaryToolSession,
    input: CreateOrGetTermInput,
    enableCharacterGenderWarnings = false,
): unknown {
    if (!hasSourceQueryForCreate(session, input.source_text, input.source_language)) {
        return noQuerySnapshotOutput('create_or_get_term 调用前必须先针对该候选词调用 query_glossary_terms，并至少包含 exact / alias / compound 查询。')
    }

    const normalizedSourceKey = normalizeSourceText(input.source_text)
    const existingTerm = state.terms.find(term => term.source_language === input.source_language && normalizeSourceText(term.source_text) === normalizedSourceKey && (term.status === 'active' || term.status === 'merged'))

    if (existingTerm?.status === 'merged') {
        const targetTerm = existingTerm.merged_into
            ? state.terms.find(term => term.term_id === existingTerm.merged_into)
            : undefined

        return {
            ok: false,
            code: 'term_merged',
            message: '命中 merged Term，请查询 merged_into 指向的目标 Term 后再继续。',
            term_id: existingTerm.term_id,
            merged_into: existingTerm.merged_into,
            retry_required: true,
            retry_tool: 'query_glossary_terms',
            retry_query: createTermRetryQuery(targetTerm ?? existingTerm),
        }
    }

    if (existingTerm) {
        return {
            ok: false,
            code: 'exact_duplicate_term',
            message: '候选词与已有 active Term 的 source_text 精确重复，请复用已有 Term，不要创建新 Term。',
            duplicate_term_id: existingTerm.term_id,
            term: formatTermForAgent(existingTerm),
            normalized_source_key: `${input.source_language}:${normalizedSourceKey}`,
            retry_required: true,
            retry_tool: 'query_glossary_terms',
            retry_query: createTermRetryQuery(existingTerm),
        }
    }

    const exactDuplicateTerm = findExactDuplicateTerm(state, input)

    if (exactDuplicateTerm) {
        return {
            ok: false,
            code: 'exact_duplicate_term',
            message: '候选词或 alias 与已有 active/merged Term 的 source_text 或 aliases 规范化后重合，请复用已有 Term，不要创建新 Term。',
            duplicate_term_id: exactDuplicateTerm.term_id,
            term: formatTermForAgent(exactDuplicateTerm),
            retry_required: true,
            retry_tool: 'query_glossary_terms',
            retry_query: createTermRetryQuery(exactDuplicateTerm),
        }
    }

    const possibleDuplicateTerms = findPossibleDuplicateTerms(state, input)
        .filter(term => !(input.confirmed_distinct_from_term_ids ?? []).includes(term.term_id))

    if (possibleDuplicateTerms.length > 0) {
        return {
            ok: false,
            code: 'possible_duplicate_term',
            message: '候选词与已有 Term 存在 compound/fuzzy 疑似重复。请优先复用已有 Term 或提交合并申请；如确认不是同一术语，可携带 confirmed_distinct_from_term_ids 和 distinct_reason 再次创建。',
            confirmation_required: true,
            duplicate_term_ids: possibleDuplicateTerms.map(term => term.term_id),
            terms: possibleDuplicateTerms.map(formatTermForAgent),
            retry_required: true,
            retry_tool: 'query_glossary_terms',
            retry_query: createPossibleDuplicateRetryQuery(input, possibleDuplicateTerms),
        }
    }

    const now = new Date().toISOString()
    const termId = createNextTermId(state)
    const aliasState = createAliasState(termId, input.aliases)
    const term: GlossaryTerm = {
        term_id: termId,
        source_text: input.source_text,
        source_language: input.source_language,
        term_type: input.term_type,
        ...aliasState,
        status: 'active',
        merged_into: null,
        entry_ids: [],
        created_at: now,
        updated_at: now,
        revision: 1,
        created_by: input.created_by,
        created_from: input.created_from,
    }

    state.terms.push(term)
    state.meta.updated_at = now
    recordQueriedTerm(session, term)
    refreshSessionSnapshots(session, state)

    return {
        ok: true,
        action: 'created',
        term: formatTermForAgent(term),
        normalized_source_key: `${input.source_language}:${normalizedSourceKey}`,
        ...formatNextSteps(collectTermNextSteps(state, term)),
        warnings: collectAliasCoverageWarnings(state, term, input.aliases),
        ...formatCharacterContextWarnings(enableCharacterGenderWarnings ? collectCharacterContextWarnings({
            terms: state.terms,
            entries: state.entries,
            termIds: new Set([term.term_id]),
        }) : []),
        retry_required: false,
    }
}

function runCreateOrGetTerms (
    state: GlossaryState,
    session: GlossaryToolSession,
    input: CreateOrGetTermsInput,
): unknown {
    const results = input.items.map((item, index) => {
        const { client_id: clientId, ...termInput } = item
        const result = runCreateOrGetTerm(state, session, termInput)

        return decorateBatchResult(result, index, clientId)
    })

    return batchOutput(results)
}

async function runCreateTermMergeProposal (
    root: string,
    manualTransFile: string,
    evidenceScope: EvidenceScope|undefined,
    state: GlossaryState,
    session: GlossaryToolSession,
    input: CreateTermMergeProposalInput,
): Promise<unknown> {
    const pairKey = termPairKey(input.source_term_id, input.target_term_id)

    if (!session.queriedMergeProposalPairs.has(pairKey)) {
        return {
            ok: false,
            code: 'no_merge_proposal_query',
            message: 'create_term_merge_proposal 调用前必须先针对同一 source/target term pair 调用 query_term_merge_proposals。',
            retry_required: true,
            retry_tool: 'query_term_merge_proposals',
            retry_query: {
                source_term_id: input.source_term_id,
                target_term_id: input.target_term_id,
                statuses: ['candidate', 'approved', 'applied'],
                include_terms: true,
                limit: 20,
            },
        }
    }

    const sourceTerm = state.terms.find(term => term.term_id === input.source_term_id)
    const targetTerm = state.terms.find(term => term.term_id === input.target_term_id)
    const termErrors = validateMergeProposalTerms(sourceTerm, targetTerm)

    if (termErrors.length > 0) {
        return validationErrorOutput(termErrors)
    }

    const duplicateProposal = state.merge_proposals.find(proposal => (
        termPairKey(proposal.source_term_id, proposal.target_term_id) === pairKey
        && ['candidate', 'approved', 'applied'].includes(proposal.status)
    ))

    if (duplicateProposal) {
        return {
            ok: false,
            code: 'duplicate_merge_proposal',
            message: '该 Term pair 已存在未关闭的合并申请，请复用已有申请，不要重复创建。',
            proposal: duplicateProposal,
            retry_required: false,
        }
    }

    const evidenceErrors = validateExistingEvidenceIdsForProposal(input.existing_evidence_ids, state, session)

    if (evidenceErrors.length > 0) {
        return validationErrorOutput(evidenceErrors)
    }

    let normalizedInputEvidence: NormalizedEvidencePayload[] = []
    if (input.evidence.length > 0) {
        const manualTrans = await loadManualTransData(root, manualTransFile)
        const normalizedEvidence = await normalizeEvidencePayloads(root, input.evidence, manualTrans, evidenceScope, session.allowedEvidenceRanges)

        if (normalizedEvidence.errors.length > 0) {
            return validationErrorOutput([...evidenceErrors, ...normalizedEvidence.errors])
        }

        normalizedInputEvidence = normalizedEvidence.evidence
    }

    const now = new Date().toISOString()
    const proposal: GlossaryMergeProposal = {
        proposal_id: createNextMergeProposalId(state),
        source_term_id: input.source_term_id,
        target_term_id: input.target_term_id,
        status: 'candidate',
        reason: input.reason,
        evidence: normalizedInputEvidence.map(item => ({
            ...item,
            source_ref: {
                source_file_id: item.source_ref.source_file_id,
                key_index: item.source_ref.key_index,
                key_hash: item.source_ref.key_hash,
                span: item.source_ref.span,
            },
        })),
        existing_evidence_ids: uniqueStrings(input.existing_evidence_ids),
        created_by: input.created_by,
        created_from: input.created_from,
        revision: 1,
        created_at: now,
        updated_at: now,
    }

    state.merge_proposals.push(proposal)
    state.meta.updated_at = now

    return {
        ok: true,
        action: 'created',
        proposal,
        retry_required: false,
    }
}

function runUpdateTermMetadata (
    state: GlossaryState,
    session: GlossaryToolSession,
    input: UpdateTermMetadataInput,
    enableCharacterGenderWarnings = false,
): unknown {
    if (!session.queriedTermIds.has(input.term_id)) {
        return noQuerySnapshotOutput('update_term_metadata 调用前必须先通过 query_glossary_terms 查询该 Term。', input.term_id)
    }

    const term = state.terms.find(candidate => candidate.term_id === input.term_id)

    if (!term) {
        return validationErrorOutput([
            {
                field: 'term_id',
                message: `${input.term_id} 不存在。`,
            },
        ])
    }

    if (term.revision !== input.expected_term_revision) {
        return {
            ok: false,
            code: 'term_revision_conflict',
            message: 'expected_term_revision 与当前 Term revision 不一致，请重新查询后再判断。',
            term_id: input.term_id,
            retry_required: true,
            retry_tool: 'query_glossary_terms',
            retry_query: createTermRetryQuery(term),
        }
    }

    const queriedRevision = session.queriedTermRevisions.get(input.term_id)

    if (queriedRevision !== undefined && queriedRevision !== term.revision) {
        return {
            ok: false,
            code: 'stale_query_snapshot',
            message: '目标 Term 自上次查询后已发生变化，请重新查询该 Term 后再判断。',
            term_id: input.term_id,
            retry_required: true,
            retry_tool: 'query_glossary_terms',
            retry_query: createTermRetryQuery(term),
        }
    }

    const now = new Date().toISOString()

    let nextSteps: string[] = []
    let aliasCoverageWarningTexts: string[] = []

    if (input.operation === 'add_aliases') {
        const aliasErrors: ValidationError[] = []
        validateAliases('aliases_to_add', input.aliases_to_add ?? [], term.source_language, aliasErrors)
        if (aliasErrors.length > 0) {
            return validationOutput(aliasErrors)
        }

        const existingAliases = new Set(getAliasTexts(term).map(alias => alias.trim()).filter(Boolean))
        addAliasesToTerm(term, input.aliases_to_add ?? [])
        const addedAliases = (input.aliases_to_add ?? []).map(alias => alias.trim()).filter(alias => alias && !existingAliases.has(alias))
        aliasCoverageWarningTexts = collectAliasCoverageWarnings(state, term, addedAliases)
    } else if (input.operation === 'change_term_type') {
        if (!input.term_type) {
            return validationErrorOutput([{ field: 'term_type', message: 'change_term_type 必须提供 term_type。' }])
        }
        term.term_type = input.term_type
        normalizeTermGenderPresentations(state, term)
        nextSteps = collectTermNextSteps(state, term)
    } else if (input.operation === 'deprecate_term') {
        term.status = 'deprecated'
        clearRejectedReason(term)
    } else if (input.operation === 'reject_term') {
        rejectTerm(term, input.rejected_reason ?? undefined)
    } else if (input.operation === 'merge_term') {
        if (!input.merged_into || !session.queriedTermIds.has(input.merged_into)) {
            return {
                ok: false,
                code: 'invalid_merge_target',
                message: 'merge_term 的 merged_into 目标 Term 必须已被当前 agent 查询过。',
                term_id: input.term_id,
                retry_required: true,
                retry_tool: 'query_glossary_terms',
                retry_query: {},
            }
        }

        term.status = 'merged'
        term.merged_into = input.merged_into
        clearRejectedReason(term)
    } else if (input.operation === 'append_gender_presentation') {
        const presentation = input.gender_presentation
        if (!presentation) {
            return validationErrorOutput([{ field: 'gender_presentation', message: 'append_gender_presentation 必须提供 gender_presentation。' }])
        }
        const genderErrors = validateGenderPresentationBinding(state, term, presentation, 'gender_presentation')
        if (genderErrors.length > 0) {
            return validationErrorOutput(genderErrors)
        }
        term.gender_presentations = [...(term.gender_presentations ?? []), presentation]
    }

    term.revision += 1
    term.updated_at = now
    markTermUpdatedFromBatch(term, readCreatedBatchId(input.updated_from), readUpdatedMetadataFields(input))
    state.meta.updated_at = now
    recordQueriedTerm(session, term)
    refreshSessionSnapshots(session, state)

    return {
        ok: true,
        operation: input.operation,
        term_id: term.term_id,
        term_revision: term.revision,
        term: formatTermForAgent(term),
        ...formatNextSteps(nextSteps),
        warnings: aliasCoverageWarningTexts,
        ...formatCharacterContextWarnings(enableCharacterGenderWarnings ? collectCharacterContextWarnings({
            terms: state.terms,
            entries: state.entries,
            termIds: new Set([term.term_id]),
        }) : []),
        retry_required: false,
    }
}

async function runUpdateTermEntries (
    root: string,
    manualTransFile: string,
    evidenceScope: EvidenceScope|undefined,
    state: GlossaryState,
    session: GlossaryToolSession,
    input: UpdateTermEntriesInput,
    enableTargetTermRevisionCheck: boolean,
): Promise<unknown> {
    if (!session.queriedTermIds.has(input.term_id)) {
        return noQuerySnapshotOutput('update_term_entries 调用前必须先通过 query_glossary_terms 查询该 Term。', input.term_id)
    }

    const term = state.terms.find(candidate => candidate.term_id === input.term_id)

    if (!term) {
        return validationErrorOutput([{ field: 'term_id', message: `${input.term_id} 不存在。` }])
    }

    const manualTrans = input.evidence.length > 0 ? await loadManualTransData(root, manualTransFile) : null
    const output = await runAppendTermEntry(root, state, session, input, term, manualTrans, evidenceScope, enableTargetTermRevisionCheck)
    refreshSessionSnapshots(session, state)
    return output
}

async function runAppendTermEntriesBatch (
    root: string,
    manualTransFile: string,
    evidenceScope: EvidenceScope|undefined,
    state: GlossaryState,
    session: GlossaryToolSession,
    input: AppendTermEntriesBatchInput,
    enableTargetTermRevisionCheck: boolean,
): Promise<unknown> {
    const manualTrans = input.items.some(item => item.evidence.length > 0) ? await loadManualTransData(root, manualTransFile) : null
    const results = await Promise.all(input.items.map(async (item, index) => {
        const term = state.terms.find(candidate => candidate.term_id === item.term_id)
        const result = term
            ? await runAppendTermEntry(root, state, session, item, term, manualTrans, evidenceScope, enableTargetTermRevisionCheck)
            : validationErrorOutput([{ field: 'term_id', message: `${item.term_id} 不存在。` }])

        return decorateBatchResult(result, index, item.client_id)
    }))

    refreshSessionSnapshots(session, state)
    return batchOutput(results)
}

async function runAppendTermEntry (
    root: string,
    state: GlossaryState,
    session: GlossaryToolSession,
    input: UpdateTermEntriesInput,
    term: GlossaryTerm,
    manualTrans: ManualTransData|null,
    evidenceScope: EvidenceScope|undefined,
    enableTargetTermRevisionCheck: boolean,
): Promise<unknown> {
    if (!session.queriedTermIds.has(input.term_id)) {
        return noQuerySnapshotOutput('update_term_entries 调用前必须先通过 query_glossary_terms 查询该 Term。', input.term_id)
    }

    if (enableTargetTermRevisionCheck) {
        const queriedRevision = session.queriedTermRevisions.get(input.term_id)

        if (queriedRevision === undefined) {
            return noQuerySnapshotOutput('update_term_entries 调用前必须先通过 query_glossary_terms 查询该 Term。', input.term_id)
        }

        if (queriedRevision !== term.revision) {
            return {
                ok: false,
                code: 'stale_query_snapshot',
                message: '目标 Term 自上次查询后已新增 Entry 或更新 metadata，请重新查询完整 Entry 后再判断是否需要追加。',
                term_id: input.term_id,
                retry_required: true,
                retry_tool: 'query_glossary_terms',
                retry_query: createTermRetryQuery(term),
            }
        }
    }

    const requireEvidence = state.active_plan_id
        ? state.plans.find(plan => plan.plan_id === state.active_plan_id)?.term_extraction_policy.require_evidence ?? true
        : true

    if (input.operation === 'append_entry' && requireEvidence && input.evidence.length === 0 && !allowsEvidenceOptionalTranslationRule(input.entry)) {
        return validationErrorOutput([{ field: 'evidence', message: '当前规划要求 append_entry 至少提供 1 条 Evidence。' }])
    }

    const genderValidationErrors = validateNewGenderPresentationBinding(term, input)
    if (genderValidationErrors.length > 0) {
        return validationErrorOutput(genderValidationErrors)
    }

    const normalizedEvidence = manualTrans
        ? await normalizeEvidencePayloads(root, input.evidence, manualTrans, evidenceScope, session.allowedEvidenceRanges)
        : { evidence: [] as NormalizedEvidencePayload[], errors: [] as ValidationError[] }

    if (normalizedEvidence.errors.length > 0) {
        return validationErrorOutput(normalizedEvidence.errors)
    }

    const normalizedEntry = normalizeEntryForStorage(term, input.entry, 'entry')

    if (!normalizedEntry.ok) {
        return validationErrorOutput(normalizedEntry.errors)
    }

    const warnings = collectAppendTermEntryWarnings(term, normalizedEntry.entry)
    const now = new Date().toISOString()

    const entryId = createNextEntryId(state, term.term_id)
    const evidenceIds = createNextEvidenceIds(state, term.term_id, input.evidence.length)
    const entry = buildGlossaryEntryFromPayload(normalizedEntry.entry, {
        entry_id: entryId,
        term_id: term.term_id,
        evidence_ids: evidenceIds,
        created_by: input.created_by,
        created_from: input.created_from,
        revision: 1,
        created_at: now,
        updated_at: now,
    })

    state.entries.push(entry)
    state.evidence.push(...normalizedEvidence.evidence.map((item, index): GlossaryEvidence => ({
        evidence_id: evidenceIds[index],
        term_id: term.term_id,
        entry_id: entryId,
        source_ref: {
            source_file_id: item.source_ref.source_file_id,
            key_index: item.source_ref.key_index,
            key_hash: item.source_ref.key_hash ? normalizeKeyHashForStorage(item.source_ref.key_hash) : undefined,
            span: item.source_ref.span,
        },
        quote: item.quote,
        context: item.context,
        reason: item.reason,
        created_by: input.created_by,
        created_at: now,
    })))
    term.entry_ids = uniqueStrings([...term.entry_ids, entryId])
    if (input.gender_presentation) {
        term.gender_presentations = [
            ...(term.gender_presentations ?? []),
            {
                ...input.gender_presentation,
                entry_id: entryId,
            },
        ]
    }
    term.revision += 1
    term.updated_at = now
    markTermUpdatedFromBatch(term, readCreatedBatchId(input.created_from), [
        'entry_ids',
        ...(input.gender_presentation ? ['gender_presentations'] : []),
    ])
    state.meta.updated_at = now
    recordQueriedTerm(session, term)

    return {
        ok: true,
        operation: 'append_entry',
        term_id: term.term_id,
        entry_id: entryId,
        entry_revision: entry.revision,
        evidence_ids: evidenceIds,
        term_revision: term.revision,
        warnings,
        retry_required: false,
    }
}

function collectAppendTermEntryWarnings (term: GlossaryTerm, entry: EntryPayload): string[] {
    if (term.term_type !== 'character' || entry.entry_type !== 'translation_rule') {
        return []
    }

    const variants = getSourceVariants(term)
    const hasAlias = variants.some(variant => variant.index > 0)
    const selectors = entry.applicability.source_selectors ?? []

    if (!hasAlias || selectors.length !== 1 || selectors[0]?.variant_id !== 'term') {
        return []
    }

    return [
        'character term has aliases not covered by this translation_rule. If an alias independently enters translation and needs a fixed rendering, create or review an alias-specific translation_rule; do not add one without evidence or project convention.',
    ]
}

function collectTermNextSteps (state: GlossaryState, term: GlossaryTerm): string[] {
    if (!termTypesNeedingBaseTranslationRuleHint.has(term.term_type) || hasCandidateOrApprovedTranslationRuleCoverage(state, term.term_id, 'term')) {
        return []
    }

    if (term.term_type === 'repeated_phrase' && !shouldHintRepeatedPhrase(term.source_text)) {
        return []
    }

    const label = term.term_type === 'character' ? 'character base name' : `${term.term_type} base`
    return [`hint: consider translation_rule coverage for ${label} source_variant_indexes[0], or defer with reason.`]
}

const termTypesNeedingBaseTranslationRuleHint = new Set<GlossaryTerm['term_type']>([
    'character',
    'faction',
    'place',
    'title',
    'item',
    'repeated_phrase',
])

function shouldHintRepeatedPhrase (sourceText: string): boolean {
    return sourceText.normalize('NFKC').trim().length >= 4
}

function formatNextSteps (nextSteps: string[]): { next_steps?: string[] } {
    return nextSteps.length > 0 ? { next_steps: nextSteps } : {}
}

function collectAliasCoverageWarnings (state: GlossaryState, term: GlossaryTerm, aliasTexts: string[]): string[] {
    if (term.term_type !== 'character') {
        return []
    }

    const targetAliases = new Set(aliasTexts.map(alias => alias.trim()).filter(Boolean))

    if (targetAliases.size === 0) {
        return []
    }

    return getSourceVariants(term)
        .filter(variant => variant.index > 0 && targetAliases.has(variant.text))
        .filter(variant => !hasCandidateOrApprovedTranslationRuleCoverage(state, term.term_id, variant.variant_id))
        .map(variant => `character alias may need translation_rule coverage: ${variant.text} (${variant.variant_id}). If this alias independently enters translation and has a stable rendering, create an alias-specific translation_rule; otherwise report why it is deferred.`)
}

function hasCandidateOrApprovedTranslationRuleCoverage (state: GlossaryState, termId: string, variantId: string): boolean {
    return state.entries.some(entry => {
        if (entry.term_id !== termId || entry.entry_type !== 'translation_rule' || !['candidate', 'approved'].includes(entry.status)) {
            return false
        }

        return (entry.applicability.source_selectors ?? []).some(selector => selector.variant_id === variantId)
    })
}

async function runAppendReviewEntry (
    root: string,
    manualTransFile: string,
    evidenceScope: EvidenceScope|undefined,
    state: GlossaryState,
    session: GlossaryToolSession,
    input: AppendReviewEntryInput,
): Promise<unknown> {
    const window = state.review.active_window

    if (!window || window.review_window_id !== input.review_window_id) {
        return {
            ok: false,
            code: 'no_active_review_window',
            message: `当前没有 active review window ${input.review_window_id}。`,
            retry_required: false,
        }
    }

    const term = state.terms.find(item => item.term_id === input.term_id)

    if (!term) {
        return validationErrorOutput([{ field: 'term_id', message: `${input.term_id} 不存在。` }])
    }

    if (term.status !== 'active') {
        return validationErrorOutput([{ field: 'term_id', message: 'append_review_entry 只能追加到 active Term。' }])
    }

    if (!isAppendReviewTermAllowed(state, window, session, term.term_id)) {
        return validationErrorOutput([{ field: 'term_id', message: 'append_review_entry 只能追加到当前 review window 牵连、pending action 涉及或本次 review 查询/检索过的 Term。' }])
    }

    const existingEvidence = input.evidence_ids.map((evidenceId, index) => {
        const item = state.evidence.find(candidate => candidate.evidence_id === evidenceId)

        if (!item) {
            return {
                ok: false as const,
                error: { field: `evidence_ids[${index}]`, message: `${evidenceId} 不存在。` },
            }
        }

        if (item.term_id !== term.term_id) {
            return {
                ok: false as const,
                error: { field: `evidence_ids[${index}]`, message: `${evidenceId} 不属于 Term ${term.term_id}。` },
            }
        }

        return { ok: true as const, evidence: item }
    })
    const evidenceIdErrors = existingEvidence
        .filter((item): item is { ok: false, error: ValidationError } => !item.ok)
        .map(item => item.error)

    if (evidenceIdErrors.length > 0) {
        return validationErrorOutput(evidenceIdErrors)
    }

    if (input.evidence.length > 0) {
        const manualTrans = await loadManualTransData(root, manualTransFile)
        const normalizedEvidence = await normalizeEvidencePayloads(root, input.evidence, manualTrans, evidenceScope, session.allowedEvidenceRanges)

        if (normalizedEvidence.errors.length > 0) {
            return validationErrorOutput(normalizedEvidence.errors)
        }

        input = {
            ...input,
            evidence: normalizedEvidence.evidence as unknown as EvidencePayload[],
        }
    }

    const entryPayload = buildAppendReviewEntryPayload(input)
    const normalizedEntry = normalizeEntryForStorage(term, entryPayload, 'entry')

    if (!normalizedEntry.ok) {
        return validationErrorOutput(normalizedEntry.errors)
    }

    const action: ReviewEntryAction = {
        action_id: createNextReviewActionId(window.pending_entry_actions.length + 1),
        operation: 'append',
        term_id: term.term_id,
        entry: {
            entry_type: normalizedEntry.entry.entry_type,
            basis: normalizedEntry.entry.basis,
            content: normalizedEntry.entry.content,
            target: normalizedEntry.entry.target,
            applicability: normalizedEntry.entry.applicability,
            policy: normalizedEntry.entry.policy,
        },
        evidence_ids: input.evidence_ids,
        evidence: input.evidence,
        reason: input.reason,
    }

    window.pending_entry_actions.push(action)
    state.meta.updated_at = new Date().toISOString()

    return {
        ok: true,
        review_window_id: window.review_window_id,
        accepted_actions: 1,
        pending_entry_action_count: window.pending_entry_actions.length,
        retry_required: false,
    }
}

function isAppendReviewTermAllowed (
    state: GlossaryState,
    window: GlossaryReviewWindow,
    session: GlossaryToolSession,
    termId: string,
): boolean {
    return collectAppendReviewTermIds(state, window).has(termId) || session.queriedTermIds.has(termId)
}

function collectAppendReviewTermIds (state: GlossaryState, window: GlossaryReviewWindow): Set<string> {
    const batchIds = new Set(window.completed_batch_numbers.map(formatBatchId))
    const termIds = new Set<string>()

    for (const term of state.terms) {
        if (batchIds.has(readCreatedBatchId(term.created_from)) || hasTermUpdatedFromAnyBatch(term, batchIds)) {
            termIds.add(term.term_id)
        }
    }

    for (const entry of state.entries) {
        if (batchIds.has(readCreatedBatchId(entry.created_from))) {
            termIds.add(entry.term_id)
        }
    }

    for (const action of window.pending_entry_actions) {
        const entry = action.entry_id ? state.entries.find(item => item.entry_id === action.entry_id) : undefined
        if (entry) {
            termIds.add(entry.term_id)
        }
        if (action.term_id) {
            termIds.add(action.term_id)
        }
        if (action.target_term_id) {
            termIds.add(action.target_term_id)
        }
    }

    for (const action of window.pending_term_actions) {
        termIds.add(action.term_id)
        if (action.target_term_id) {
            termIds.add(action.target_term_id)
        }
    }

    return termIds
}

function buildAppendReviewEntryPayload (input: AppendReviewEntryInput): EntryPayload {
    const isTranslationRule = input.entry_type === 'translation_rule'

    return {
        entry_type: input.entry_type,
        ...(isTranslationRule && input.basis ? { basis: input.basis } : {}),
        content: {
            summary: input.summary,
            description: input.description,
        },
        ...(isTranslationRule ? {
            target: {
                target_language: DEFAULT_TARGET_LANGUAGE,
                preferred_translation: input.preferred_translation ?? undefined,
                alternative_translations: input.alternative_translations ?? [],
                forbidden_translations: input.forbidden_translations ?? [],
            },
        } : {}),
        applicability: {
            applies_to: input.applies_to,
            ...(input.applies_when ? { applies_when: input.applies_when } : {}),
            ...(input.does_not_apply_when ? { does_not_apply_when: input.does_not_apply_when } : {}),
            ...(isTranslationRule && input.source_variant_indexes ? { source_variant_indexes: input.source_variant_indexes } : {}),
        },
        policy: {
            strength: input.policy_strength,
            ...(input.requires_context_check !== null ? { requires_context_check: input.requires_context_check } : {}),
            ...(input.notes ? { notes: input.notes } : {}),
        },
        status: 'candidate',
    }
}

function runReviewEntriesBatch (state: GlossaryState, input: ReviewEntriesBatchInput): unknown {
    const window = getActiveReviewWindow(state, input.review_window_id)
    const errors: ValidationError[] = []
    input.actions.forEach((action, index) => {
        const entry = state.entries.find(item => item.entry_id === action.entry_id)

        if ((action.operation === 'move_to_term' || action.operation === 'move_to_term_and_approve') && action.target_term_id && !state.terms.some(term => term.term_id === action.target_term_id)) {
            errors.push({ field: `actions[${index}].target_term_id`, message: `target Term ${action.target_term_id} 不存在。` })
        }

        if (!action.revised_entry || !entry) {
            return
        }

        const revisedEntry = action.revised_entry as NonNullable<ReviewEntryAction['revised_entry']>
        const targetTerm = action.target_term_id
            ? state.terms.find(term => term.term_id === action.target_term_id)
            : state.terms.find(term => term.term_id === entry.term_id)
        const sourceTerm = state.terms.find(term => term.term_id === entry.term_id)
        const entryPreview = structuredClone(entry) as GlossaryEntry

        if (
            (action.operation === 'move_to_term' || action.operation === 'move_to_term_and_approve')
            && targetTerm
            && sourceTerm
            && !Array.isArray(revisedEntry.applicability?.source_variant_indexes)
        ) {
            remapEntrySourceSelectorsForMove(entryPreview, sourceTerm, targetTerm)
        }

        const effectiveRevisedEntry = targetTerm
            ? withFallbackSourceVariantIndexes(entryPreview, revisedEntry, targetTerm)
            : revisedEntry
        const preview = buildRevisedEntryPreview(entryPreview, effectiveRevisedEntry, targetTerm, `actions[${index}].revised_entry`)
        if (!preview.ok) {
            errors.push(...preview.errors)
        }
    })

    if (errors.length > 0) {
        return validationErrorOutput(errors)
    }

    const now = new Date().toISOString()
    const actions: ReviewEntryAction[] = input.actions.map((action, index) => ({
        action_id: createNextReviewActionId(window.pending_entry_actions.length + index + 1),
        operation: action.operation,
        entry_id: action.entry_id,
        expected_entry_revision: action.expected_entry_revision ?? undefined,
        target_entry_id: action.target_entry_id ?? undefined,
        target_term_id: action.target_term_id ?? undefined,
        revised_entry: action.revised_entry ?? undefined,
        reason: action.reason,
    }))
    const actionResults = actions.flatMap(action => {
        if (action.operation !== 'move_to_term' && action.operation !== 'move_to_term_and_approve') {
            return []
        }

        const entry = state.entries.find(item => item.entry_id === action.entry_id)
        return [{
            action_id: action.action_id,
            operation: action.operation,
            moved: true,
            revised: !!action.revised_entry,
            status: action.operation === 'move_to_term_and_approve'
                ? 'approved'
                : entry?.status ?? 'candidate',
        }]
    })
    const warnings = collectReviewEntryActionWarnings(state, window, actions)

    window.pending_entry_actions.push(...actions)
    state.meta.updated_at = now

    return {
        ok: true,
        review_window_id: window.review_window_id,
        accepted_actions: actions.length,
        pending_entry_action_count: window.pending_entry_actions.length,
        ...(actionResults.length > 0 ? { action_results: actionResults } : {}),
        warnings,
        retry_required: false,
    }
}

function collectReviewEntryActionWarnings (
    state: GlossaryState,
    window: GlossaryReviewWindow,
    newActions: ReviewEntryAction[],
): unknown[] {
    const warnings: unknown[] = []
    const allActions = [...window.pending_entry_actions, ...newActions]
    const effectiveEntries = collectEffectiveEntryStates(state, allActions)

    for (const action of newActions) {
        if (action.operation === 'move_to_term') {
            const effectiveEntry = effectiveEntries.find(entry => entry.entry_id === action.entry_id)
            if (effectiveEntry?.status === 'candidate') {
                warnings.push({
                    code: 'moved_entry_remains_candidate',
                    severity: 'warning',
                    entry_id: action.entry_id,
                    term_id: effectiveEntry.term_id,
                    target_term_id: action.target_term_id,
                    message: `${action.entry_id} was moved but remains candidate; use move_to_term_and_approve when the moved Entry should be approved immediately.`,
                })
            }
        }

        if ((action.operation === 'move_to_term' || action.operation === 'move_to_term_and_approve') && !action.revised_entry) {
            const entry = state.entries.find(item => item.entry_id === action.entry_id)
            const sourceTerm = entry ? state.terms.find(item => item.term_id === entry.term_id) : undefined

            if (entry && sourceTerm && entryStillMentionsSourceTerm(entry, sourceTerm.source_text)) {
                warnings.push(`${action.entry_id} is moved without revised_entry, but its content or applicability still mentions source Term ${sourceTerm.source_text}.`)
            }
        }

        if (action.operation === 'reject' && action.entry_id) {
            const entry = state.entries.find(item => item.entry_id === action.entry_id)
            const term = entry ? state.terms.find(item => item.term_id === entry.term_id) : undefined
            if (!term || effectiveTermStatus(term, window) !== 'active') {
                continue
            }

            const hasEffectiveEntry = effectiveEntries.some(item => (
                item.term_id === term.term_id
                && (item.status === 'approved' || item.status === 'candidate')
            ))

            if (!hasEffectiveEntry) {
                warnings.push({
                    code: 'term_will_have_no_effective_entries',
                    severity: 'warning',
                    term_id: term.term_id,
                    source_text: term.source_text,
                    entry_id: action.entry_id,
                    message: `${term.term_id} will have no approved or candidate Entries after this rejection.`,
                })
            }
        }
    }

    return warnings
}

function entryStillMentionsSourceTerm (entry: GlossaryEntry, sourceText: string): boolean {
    const fields = [
        entry.content.summary,
        entry.content.description,
        entry.applicability.domain,
        ...(entry.applicability.applies_when ?? []),
        ...(entry.applicability.does_not_apply_when ?? []),
        ...(entry.applicability.source_selectors ?? []).map(selector => selector.text),
    ].filter((value): value is string => typeof value === 'string' && value.length > 0)

    return fields.some(value => value.includes(sourceText))
}

function collectEffectiveEntryStates (
    state: GlossaryState,
    actions: ReviewEntryAction[],
): Array<Pick<GlossaryEntry, 'entry_id'|'term_id'|'status'>> {
    const entries = state.entries.map(entry => ({
        entry_id: entry.entry_id,
        term_id: entry.term_id,
        status: entry.status,
    }))

    for (const action of actions) {
        if (action.operation === 'append' && action.term_id && action.entry) {
            entries.push({
                entry_id: `pending:${action.action_id}`,
                term_id: action.term_id,
                status: 'approved',
            })
            continue
        }

        const entry = entries.find(item => item.entry_id === action.entry_id)
        if (!entry) {
            continue
        }

        if (action.operation === 'approve' || action.operation === 'revise' || action.operation === 'move_to_term_and_approve') {
            entry.status = 'approved'
        } else if (action.operation === 'reject' || action.operation === 'merge_into') {
            entry.status = 'rejected'
        }

        if ((action.operation === 'move_to_term' || action.operation === 'move_to_term_and_approve') && action.target_term_id) {
            entry.term_id = action.target_term_id
        }
    }

    return entries
}

function effectiveTermStatus (term: GlossaryTerm, window: GlossaryReviewWindow): TermStatus {
    let status = term.status

    for (const action of window.pending_term_actions) {
        if (action.term_id !== term.term_id) {
            continue
        }

        if (action.operation === 'reject') {
            status = 'rejected'
        } else if (action.operation === 'deprecate') {
            status = 'deprecated'
        } else if (action.operation === 'merge_term') {
            status = 'merged'
        } else if (action.operation === 'keep_active') {
            status = 'active'
        }
    }

    return status
}

function collectActiveTermWithoutEffectiveEntryWarnings (state: GlossaryState): unknown[] {
    return state.terms
        .filter(term => term.status === 'active')
        .filter(term => !state.entries.some(entry => (
            entry.term_id === term.term_id
            && (entry.status === 'approved' || entry.status === 'candidate')
        )))
        .map(term => ({
            code: 'active_term_without_effective_entries',
            severity: 'warning',
            term_id: term.term_id,
            source_text: term.source_text,
            message: 'Active term has no approved or candidate entries; reject/deprecate it, add a justified entry, or explain defer in review summary.',
        }))
}

function collectCharacterContextWarningsForWindow (state: GlossaryState, window: GlossaryReviewWindow): unknown[] {
    return collectCharacterContextWarnings({
        terms: state.terms.map(term => applyPendingTermActionsForCharacterContext(term, window.pending_term_actions)),
        entries: state.entries,
        termIds: collectReviewWindowRelatedTermIds(state, window),
    })
}

function collectReviewWindowRelatedTermIds (state: GlossaryState, window: GlossaryReviewWindow): Set<string> {
    const batchIds = new Set(window.completed_batch_numbers.map(formatBatchId))
    const termIds = new Set<string>()

    for (const term of state.terms) {
        if (batchIds.has(readCreatedBatchId(term.created_from)) || hasTermUpdatedFromAnyBatch(term, batchIds)) {
            termIds.add(term.term_id)
        }
    }

    for (const entry of state.entries) {
        if (batchIds.has(readCreatedBatchId(entry.created_from))) {
            termIds.add(entry.term_id)
        }
    }

    for (const action of window.pending_entry_actions) {
        const entry = action.entry_id ? state.entries.find(item => item.entry_id === action.entry_id) : null

        if (entry) {
            termIds.add(entry.term_id)
        }
        if (action.term_id) {
            termIds.add(action.term_id)
        }
        if (action.target_term_id) {
            termIds.add(action.target_term_id)
        }
    }

    for (const action of window.pending_term_actions) {
        termIds.add(action.term_id)
        if (action.target_term_id) {
            termIds.add(action.target_term_id)
        }
    }

    return termIds
}

function applyPendingTermActionsForCharacterContext (term: GlossaryTerm, actions: ReviewTermAction[]): GlossaryTerm {
    let nextTerm = term

    for (const action of actions) {
        if (action.term_id !== term.term_id) {
            continue
        }

        if (nextTerm === term) {
            nextTerm = {
                ...term,
                aliases: { ...term.aliases },
                alias_order: [...term.alias_order],
                entry_ids: [...term.entry_ids],
                gender_presentations: term.gender_presentations ? [...term.gender_presentations] : undefined,
            }
        }

        if (action.operation === 'reject') {
            nextTerm.status = 'rejected'
        } else if (action.operation === 'deprecate') {
            nextTerm.status = 'deprecated'
        } else if (action.operation === 'merge_term') {
            nextTerm.status = 'merged'
        } else if (action.operation === 'keep_active') {
            nextTerm.status = 'active'
        } else if (action.operation === 'change_term_type' && action.term_type) {
            nextTerm.term_type = action.term_type
        } else if (action.operation === 'add_aliases') {
            addAliasesToTerm(nextTerm, action.aliases_to_add ?? [])
        } else if (action.operation === 'remove_invalid_aliases') {
            removeAliasesFromTermByText(nextTerm, action.aliases_to_remove ?? [])
        } else if (action.operation === 'set_gender_presentations') {
            nextTerm.gender_presentations = action.gender_presentations ?? []
        }
    }

    return nextTerm
}

function formatCharacterContextWarnings (warnings: unknown[]): { character_context_warnings?: unknown[] } {
    return warnings.length > 0 ? { character_context_warnings: warnings } : {}
}

function runReviewTermsBatch (state: GlossaryState, input: ReviewTermsBatchInput, enableCharacterGenderWarnings = false): unknown {
    const window = getActiveReviewWindow(state, input.review_window_id)
    const now = new Date().toISOString()
    const actions: ReviewTermAction[] = input.actions.map((action, index) => ({
        action_id: createNextReviewActionId(window.pending_term_actions.length + index + 1),
        operation: action.operation,
        term_id: action.term_id,
        expected_term_revision: action.expected_term_revision ?? undefined,
        target_term_id: action.target_term_id ?? undefined,
        term_type: action.term_type ?? undefined,
        aliases_to_add: action.aliases_to_add ?? undefined,
        aliases_to_remove: action.aliases_to_remove ?? undefined,
        entry_ids: action.entry_ids ?? undefined,
        gender_presentations: action.gender_presentations ?? undefined,
        rejected_reason: action.rejected_reason ?? undefined,
        reason: action.reason,
    }))
    const warnings = collectReviewTermActionWarnings(state, window, actions)
    const characterContextWarnings = enableCharacterGenderWarnings
        ? collectCharacterContextWarnings({
            terms: state.terms.map(term => applyPendingTermActionsForCharacterContext(term, [...window.pending_term_actions, ...actions])),
            entries: state.entries,
            termIds: new Set(actions.map(action => action.term_id)),
        })
        : []

    window.pending_term_actions.push(...actions)
    state.meta.updated_at = now

    return {
        ok: true,
        review_window_id: window.review_window_id,
        accepted_actions: actions.length,
        pending_term_action_count: window.pending_term_actions.length,
        warnings,
        ...(characterContextWarnings.length > 0 ? { character_context_warnings: characterContextWarnings } : {}),
        retry_required: false,
    }
}

function collectReviewTermActionWarnings (
    state: GlossaryState,
    window: GlossaryReviewWindow,
    newActions: ReviewTermAction[],
): unknown[] {
    const warnings: unknown[] = []
    const allEntryActions = window.pending_entry_actions

    for (const action of newActions) {
        if (action.operation !== 'reject' && action.operation !== 'deprecate' && action.operation !== 'merge_term') {
            continue
        }

        const term = state.terms.find(item => item.term_id === action.term_id)
        if (!term) {
            continue
        }

        const unhandledEntries = state.entries.filter(entry => (
            entry.term_id === term.term_id
            && isSemanticEntry(entry)
            && (entry.status === 'approved' || entry.status === 'candidate')
            && !isEntryExplicitlyHandledAway(entry.entry_id, allEntryActions)
        ))

        if (unhandledEntries.length === 0) {
            continue
        }

        warnings.push({
            code: 'term_action_discards_semantic_entries',
            severity: 'warning',
            term_id: term.term_id,
            source_text: term.source_text,
            operation: action.operation,
            target_term_id: action.target_term_id,
            entries: unhandledEntries.map(entry => ({
                entry_id: entry.entry_id,
                entry_type: entry.entry_type,
                status: entry.status,
                summary: entry.content.summary,
                evidence_count: entry.evidence_ids.length,
            })),
            message: `${action.operation} on ${term.term_id} may discard fact/style/continuity Entries; explicitly reject, merge, revise, or move them before clearing the Term.`,
        })
    }

    return warnings
}

function isSemanticEntry (entry: Pick<GlossaryEntry, 'entry_type'>): boolean {
    return entry.entry_type === 'fact' || entry.entry_type === 'style' || entry.entry_type === 'continuity'
}

function isEntryExplicitlyHandledAway (entryId: string, actions: ReviewEntryAction[]): boolean {
    return actions.some(action => (
        action.entry_id === entryId
        && (
            action.operation === 'reject'
            || action.operation === 'merge_into'
            || action.operation === 'move_to_term'
            || action.operation === 'move_to_term_and_approve'
        )
    ))
}

function validateListReviewCandidatesInput (input: unknown): { ok: true, value: ListReviewCandidatesInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<ListReviewCandidatesInput>(looseListReviewCandidatesSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    return {
        ok: true,
        value: {
            review_window_id: parsed.value.review_window_id ?? null,
            include_terms: parsed.value.include_terms,
            include_entries: parsed.value.include_entries,
            include_evidence: parsed.value.include_evidence,
            limit: parsed.value.limit,
        },
    }
}

function validateCheckTranslationRuleCoverageInput (input: unknown): { ok: true, value: CheckTranslationRuleCoverageInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<CheckTranslationRuleCoverageInput>(looseCheckTranslationRuleCoverageSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    return {
        ok: true,
        value: {
            review_window_id: parsed.value.review_window_id,
            term_ids: parsed.value.term_ids ?? null,
            term_types: parsed.value.term_types ?? null,
            include_rejected_candidates: parsed.value.include_rejected_candidates,
        },
    }
}

function validateAppendReviewEntryInput (input: unknown): { ok: true, value: AppendReviewEntryInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<AppendReviewEntryInput>(looseAppendReviewEntrySchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const errors: ValidationError[] = []
    const value = parsed.value
    const isTranslationRule = value.entry_type === 'translation_rule'
    const hasTranslationFields = value.basis !== null && value.basis !== undefined
        || value.source_variant_indexes !== null && value.source_variant_indexes !== undefined
        || value.preferred_translation !== null && value.preferred_translation !== undefined
        || value.alternative_translations !== null && value.alternative_translations !== undefined
        || value.forbidden_translations !== null && value.forbidden_translations !== undefined

    if (value.evidence.length === 0 && value.evidence_ids.length === 0 && !allowsEvidenceOptionalTranslationRule(value)) {
        errors.push({ field: 'evidence', message: 'append_review_entry 必须提供 evidence 或 evidence_ids。' })
    }

    if (isTranslationRule) {
        if (!value.basis) {
            errors.push({ field: 'basis', message: 'translation_rule 必须提供 basis。' })
        }
        if (!value.source_variant_indexes || value.source_variant_indexes.length === 0) {
            errors.push({ field: 'source_variant_indexes', message: 'translation_rule 必须提供 source_variant_indexes。' })
        }
        if (!hasFlatTranslationTargetConstraint(value)) {
            errors.push({ field: 'target', message: 'translation_rule 必须提供 target，且至少包含 preferred_translation、alternative_translations 或 forbidden_translations。' })
        }
    } else if (hasTranslationFields) {
        errors.push({ field: '(root)', message: `${value.entry_type} Entry 不允许包含 translation_rule 专用字段。` })
    }

    if (errors.length > 0) {
        return { ok: false, errors }
    }

    return {
        ok: true,
        value: {
            ...value,
            applies_when: value.applies_when ?? null,
            does_not_apply_when: value.does_not_apply_when ?? null,
            requires_context_check: value.requires_context_check ?? null,
            notes: value.notes ?? null,
            basis: value.basis ?? null,
            source_variant_indexes: value.source_variant_indexes ?? null,
            preferred_translation: value.preferred_translation ?? null,
            alternative_translations: value.alternative_translations ?? null,
            forbidden_translations: value.forbidden_translations ?? null,
        },
    }
}

function validateReviewEntriesBatchInput (input: unknown): { ok: true, value: ReviewEntriesBatchInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<ReviewEntriesBatchInput>(looseReviewEntriesBatchSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const errors: ValidationError[] = []
    parsed.value.actions.forEach((action, index) => {
        if (action.operation === 'merge_into' && !action.target_entry_id) {
            errors.push({ field: `actions[${index}].target_entry_id`, message: 'merge_into 必须提供 target_entry_id。' })
        }

        if ((action.operation === 'move_to_term' || action.operation === 'move_to_term_and_approve') && !action.target_term_id) {
            errors.push({ field: `actions[${index}].target_term_id`, message: `${action.operation} 必须提供 target_term_id。` })
        }

        if (action.operation === 'revise' && !action.revised_entry) {
            errors.push({ field: `actions[${index}].revised_entry`, message: 'revise 必须提供 revised_entry。' })
        }

        const revisedEntryType = action.revised_entry?.entry_type
        if ((revisedEntryType === 'fact' || revisedEntryType === 'style' || revisedEntryType === 'continuity') && action.revised_entry?.target) {
            errors.push({ field: `actions[${index}].revised_entry.target`, message: `${revisedEntryType} Entry 不允许包含 target。` })
        }
        if (revisedEntryType && revisedEntryType !== 'translation_rule' && action.revised_entry?.basis) {
            errors.push({ field: `actions[${index}].revised_entry.basis`, message: 'basis 只允许用于 translation_rule Entry。' })
        }
    })

    return errors.length > 0 ? { ok: false, errors } : { ok: true, value: parsed.value }
}

function validateReviewTermsBatchInput (input: unknown): { ok: true, value: ReviewTermsBatchInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<ReviewTermsBatchInput>(looseReviewTermsBatchSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const errors: ValidationError[] = []

    parsed.value.actions.forEach((action, index) => {
        if (action.operation !== 'reject' && action.rejected_reason !== undefined) {
            errors.push({ field: `actions[${index}].rejected_reason`, message: `${action.operation} 不接受 rejected_reason。` })
        }

        if (action.operation === 'reject' && !action.rejected_reason) {
            errors.push({ field: `actions[${index}].rejected_reason`, message: 'reject 必须提供 rejected_reason。' })
        }

        if ((action.operation === 'merge_term' || action.operation === 'move_entries') && !action.target_term_id) {
            errors.push({ field: `actions[${index}].target_term_id`, message: `${action.operation} 必须提供 target_term_id。` })
        }

        if (action.operation === 'change_term_type' && !action.term_type) {
            errors.push({ field: `actions[${index}].term_type`, message: 'change_term_type 必须提供 term_type。' })
        }

        if (action.operation === 'add_aliases') {
            const aliases = action.aliases_to_add ?? []
            if (aliases.length === 0) {
                errors.push({ field: `actions[${index}].aliases_to_add`, message: 'add_aliases 必须提供 aliases_to_add。' })
            }
        }

        if (action.operation === 'remove_invalid_aliases' && (action.aliases_to_remove ?? []).length === 0) {
            errors.push({ field: `actions[${index}].aliases_to_remove`, message: 'remove_invalid_aliases 必须提供 aliases_to_remove。' })
        }

        if (action.operation === 'move_entries' && (action.entry_ids ?? []).length === 0) {
            errors.push({ field: `actions[${index}].entry_ids`, message: 'move_entries 必须提供 entry_ids。' })
        }

        if (action.operation === 'set_gender_presentations' && !Array.isArray(action.gender_presentations)) {
            errors.push({ field: `actions[${index}].gender_presentations`, message: 'set_gender_presentations 必须提供 gender_presentations 数组。' })
        }
    })

    return errors.length > 0 ? { ok: false, errors } : { ok: true, value: parsed.value }
}

function validateSubmitGlossaryWorkerBatchInput (
    input: unknown,
    options: CreateGlossaryToolsOptions,
): { ok: true, value: SubmitGlossaryWorkerBatchInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<SubmitGlossaryWorkerBatchInput>(looseSubmitGlossaryWorkerBatchSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const value = parsed.value
    const errors: ValidationError[] = []

    if (options.workerBatchId && value.batch_id !== options.workerBatchId) {
        errors.push({ field: 'batch_id', message: `batch_id must match current worker batch ${options.workerBatchId}.` })
    }

    if (options.workerBatchNumber !== undefined && value.batch_number !== options.workerBatchNumber) {
        errors.push({ field: 'batch_number', message: `batch_number must match current worker batch ${options.workerBatchNumber}.` })
    }

    if (options.requireWorkerBatchRead && !options.requireWorkerBatchRead()) {
        errors.push({ field: '(root)', message: 'submit_glossary_worker_batch requires a successful read_key_range call for the current batch.' })
    }

    errors.push(...validateDeferredNotesCount(value.deferred_notes, options.deferredNotesMaxItems ?? null))
    errors.push(...validateSubmitSummaryLength(value.summary, options.summaryMaxChars ?? null, 'summary'))
    value.deferred_notes.forEach((note, index) => {
        errors.push(...validateSubmitSummaryLength(note, options.summaryMaxChars ?? null, `deferred_notes[${index}]`))
    })

    return errors.length > 0 ? { ok: false, errors } : { ok: true, value }
}

async function validateSubmitGlossaryReviewInput (
    glossaryStateRoot: string,
    session: GlossaryToolSession,
    input: unknown,
    limits: {
        summaryMaxChars: number|null
        deferredNotesMaxItems: number|null
    },
): Promise<{ ok: true, value: SubmitGlossaryReviewInput }|{ ok: false, errors: ValidationError[] }> {
    const parsed = parseToolInput<SubmitGlossaryReviewInput>(looseSubmitGlossaryReviewSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const value = parsed.value
    const errors: ValidationError[] = []

    errors.push(...validateDeferredNotesCount(value.deferred_notes, limits.deferredNotesMaxItems))
    errors.push(...validateSubmitSummaryLength(value.summary, limits.summaryMaxChars, 'summary'))
    value.deferred_notes.forEach((note, index) => {
        errors.push(...validateSubmitSummaryLength(note, limits.summaryMaxChars, `deferred_notes[${index}]`))
    })

    if (!session.listedReviewWindowIds.has(value.review_window_id)) {
        errors.push({ field: 'review_window_id', message: 'submit_glossary_review requires a successful list_review_candidates call for this review_window_id.' })
    }

    if (!session.checkedCoverageReviewWindowIds.has(value.review_window_id)) {
        errors.push({ field: 'review_window_id', message: 'submit_glossary_review requires a successful check_translation_rule_coverage call for this review_window_id.' })
    }

    await readGlossaryState(glossaryStateRoot, state => {
        const window = state.review.active_window

        if (!window || window.review_window_id !== value.review_window_id) {
            errors.push({ field: 'review_window_id', message: `当前没有 active review window ${value.review_window_id}。` })
            return
        }

        const hasPendingActions = window.pending_entry_actions.length > 0 || window.pending_term_actions.length > 0
        if (!hasPendingActions && value.deferred_notes.length === 0) {
            errors.push({
                field: 'deferred_notes',
                message: 'Review has no pending actions; deferred_notes must explain the no-action or defer reason.',
            })
        }
    })

    return errors.length > 0 ? { ok: false, errors } : { ok: true, value }
}

function validateSubmitSummaryLength (value: string, maxChars: number|null, field: string): ValidationError[] {
    if (maxChars === null || value.length <= maxChars) {
        return []
    }

    return [{
        field,
        message: `${field} exceeds configured glossarySubmitSummaryMaxChars (${maxChars}).`,
    }]
}

function validateDeferredNotesCount (notes: string[], maxItems: number|null): ValidationError[] {
    if (maxItems === null || notes.length <= maxItems) {
        return []
    }

    return [{
        field: 'deferred_notes',
        message: `deferred_notes exceeds configured glossarySubmitDeferredNotesMaxItems (${maxItems}).`,
    }]
}

function validateQueryGlossaryTermsInput (input: unknown): { ok: true, value: ValidatedQueryGlossaryTermsInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<QueryGlossaryTermsInput>(looseQueryGlossaryTermsSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const regexErrors = validateRegexQueries(parsed.value.queries)
    return regexErrors.length > 0
        ? { ok: false, errors: regexErrors }
        : {
            ok: true,
            value: {
                ...parsed.value,
                entry_statuses: parsed.value.entry_statuses ?? null,
                term_statuses: parsed.value.term_statuses ?? null,
                include_rejected_details: parsed.value.include_rejected_details,
            },
        }
}

function validateSearchGlossaryEntriesInput (input: unknown): { ok: true, value: SearchGlossaryEntriesInput }|{ ok: false, errors: ValidationError[] } {
    const errors: ValidationError[] = []
    const parsed = parseToolInput<SearchGlossaryEntriesInput>(looseSearchGlossaryEntriesSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const entryFilter = normalizeEntryFilter(parsed.value.entry_filter)
    const termFilter = normalizeTermFilter(parsed.value.term_filter)

    if (!hasEffectiveEntryFilter(entryFilter) && !hasEffectiveTermFilter(termFilter)) {
        errors.push({ field: '(root)', message: 'entry_filter 和 term_filter 至少需要一个有效过滤条件。' })
    }

    if (entryFilter?.text && entryFilter.is_regex) {
        errors.push(...validateRegexText('entry_filter.text', entryFilter.text))
    }

    if (errors.length > 0) {
        return { ok: false, errors }
    }

    return {
        ok: true,
        value: {
            entry_filter: hasEffectiveEntryFilter(entryFilter) ? entryFilter : null,
            term_filter: hasEffectiveTermFilter(termFilter) ? termFilter : null,
            include_term: parsed.value.include_term,
            include_evidence: parsed.value.include_evidence,
            limit: parsed.value.limit,
        },
    }
}

function validateEvidenceContextInput (input: unknown): { ok: true, value: EvidenceContextInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<EvidenceContextInput>(looseEvidenceContextSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const errors: ValidationError[] = []

    if (parsed.value.window_before > MAX_EVIDENCE_CONTEXT_WINDOW) {
        errors.push({ field: 'window_before', message: `window_before 必须小于或等于 ${MAX_EVIDENCE_CONTEXT_WINDOW}。` })
    }

    if (parsed.value.window_after > MAX_EVIDENCE_CONTEXT_WINDOW) {
        errors.push({ field: 'window_after', message: `window_after 必须小于或等于 ${MAX_EVIDENCE_CONTEXT_WINDOW}。` })
    }

    if (parsed.value.max_chars_per_key > MAX_CHARS_PER_KEY) {
        errors.push({ field: 'max_chars_per_key', message: `max_chars_per_key 必须小于或等于 ${MAX_CHARS_PER_KEY}。` })
    }

    return errors.length > 0 ? { ok: false, errors } : parsed
}

function validateCreateOrGetTermInput (input: unknown): { ok: true, value: CreateOrGetTermInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<CreateOrGetTermInput>(looseCreateOrGetTermSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const errors: ValidationError[] = []
    validateAliases('aliases', parsed.value.aliases, parsed.value.source_language, errors)

    const confirmedDistinctIds = parsed.value.confirmed_distinct_from_term_ids ?? []
    if (confirmedDistinctIds.length > 0 && !parsed.value.distinct_reason) {
        errors.push({ field: 'distinct_reason', message: '确认候选词与疑似重复 Term 不同必须提供 distinct_reason。' })
    }

    if (errors.length > 0) {
        return { ok: false, errors }
    }

    return {
        ok: true,
        value: {
            ...parsed.value,
            confirmed_distinct_from_term_ids: parsed.value.confirmed_distinct_from_term_ids ?? null,
            distinct_reason: parsed.value.distinct_reason ?? null,
        },
    }
}

function validateCreateOrGetTermsInput (input: unknown): { ok: true, value: CreateOrGetTermsInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<CreateOrGetTermsInput>(looseCreateOrGetTermsSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const errors: ValidationError[] = []
    const items = parsed.value.items.map((item, index) => {
        validateAliases(`items[${index}].aliases`, item.aliases, item.source_language, errors)

        const confirmedDistinctIds = item.confirmed_distinct_from_term_ids ?? []
        if (confirmedDistinctIds.length > 0 && !item.distinct_reason) {
            errors.push({ field: `items[${index}].distinct_reason`, message: '确认候选词与疑似重复 Term 不同必须提供 distinct_reason。' })
        }

        return {
            ...item,
            confirmed_distinct_from_term_ids: item.confirmed_distinct_from_term_ids ?? null,
            distinct_reason: item.distinct_reason ?? null,
        }
    })

    return errors.length > 0
        ? { ok: false, errors }
        : { ok: true, value: { items } }
}

function validateQueryTermMergeProposalsInput (input: unknown): { ok: true, value: QueryTermMergeProposalsInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<QueryTermMergeProposalsInput>(looseQueryTermMergeProposalsSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const value = {
        ...parsed.value,
        source_term_id: parsed.value.source_term_id ?? null,
        target_term_id: parsed.value.target_term_id ?? null,
        statuses: parsed.value.statuses ?? null,
    }

    if (!value.source_term_id && !value.target_term_id && (!value.statuses || value.statuses.length === 0)) {
        return {
            ok: false,
            errors: [{ field: '(root)', message: 'query_term_merge_proposals 至少需要 source_term_id、target_term_id 或 statuses 之一。' }],
        }
    }

    return { ok: true, value }
}

function validateUpdateTermMetadataInput (input: unknown): { ok: true, value: UpdateTermMetadataInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<UpdateTermMetadataInput>(looseUpdateTermMetadataSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const errors: ValidationError[] = []
    const { operation, aliases_to_add: aliasesToAdd, term_type: termType, merged_into: mergedInto, gender_presentation: genderPresentation } = parsed.value
    const rawInput = isRecord(input) ? input : {}
    const hasAliasesToAdd = hasOwn(rawInput, 'aliases_to_add')
    const hasTermType = hasOwn(rawInput, 'term_type')
    const hasMergedInto = hasOwn(rawInput, 'merged_into')
    const hasGenderPresentation = hasOwn(rawInput, 'gender_presentation')
    const hasRejectedReason = hasOwn(rawInput, 'rejected_reason')

    if (operation === 'add_aliases') {
        if (!hasAliasesToAdd || !aliasesToAdd || aliasesToAdd.length === 0) {
            errors.push({ field: 'aliases_to_add', message: 'add_aliases 必须提供非空 aliases_to_add。' })
        }
    }

    if (operation === 'change_term_type' && !termType) {
        errors.push({ field: 'term_type', message: 'change_term_type 必须提供有效 term_type。' })
    }

    if (operation === 'merge_term' && !mergedInto) {
        errors.push({ field: 'merged_into', message: 'merge_term 必须提供 merged_into。' })
    }

    if (operation === 'append_gender_presentation' && !genderPresentation) {
        errors.push({ field: 'gender_presentation', message: 'append_gender_presentation 必须提供 gender_presentation。' })
    }

    if ((operation === 'deprecate_term' || operation === 'reject_term') && (hasAliasesToAdd || hasTermType || hasMergedInto || hasGenderPresentation)) {
        errors.push({ field: '(root)', message: 'deprecate_term / reject_term 不接受 aliases_to_add、term_type、merged_into、gender_presentation。' })
    }

    if (operation !== 'reject_term' && hasRejectedReason) {
        errors.push({ field: 'rejected_reason', message: `${operation} 不接受 rejected_reason。` })
    }

    if (operation === 'reject_term' && !parsed.value.rejected_reason) {
        errors.push({ field: 'rejected_reason', message: 'reject_term 必须提供 rejected_reason。' })
    }

    if (operation !== 'add_aliases' && hasAliasesToAdd) {
        errors.push({ field: 'aliases_to_add', message: `${operation} 不接受 aliases_to_add。` })
    }

    if (operation !== 'change_term_type' && hasTermType) {
        errors.push({ field: 'term_type', message: `${operation} 不接受 term_type。` })
    }

    if (operation !== 'merge_term' && hasMergedInto) {
        errors.push({ field: 'merged_into', message: `${operation} 不接受 merged_into。` })
    }

    if (operation !== 'append_gender_presentation' && hasGenderPresentation) {
        errors.push({ field: 'gender_presentation', message: `${operation} 不接受 gender_presentation。` })
    }

    if (errors.length > 0) {
        return { ok: false, errors }
    }

    return { ok: true, value: normalizeUpdateTermMetadataInput(parsed.value) }
}

function validateUpdateTermEntriesInput (input: unknown): { ok: true, value: UpdateTermEntriesInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<UpdateTermEntriesInput>(updateTermEntriesSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const errors: ValidationError[] = []
    const { operation, entry_id: entryId, expected_entry_revision: expectedEntryRevision } = parsed.value

    if (operation === 'append_entry' && (entryId || expectedEntryRevision !== null && expectedEntryRevision !== undefined)) {
        errors.push({ field: '(root)', message: 'append_entry 时 entry_id 和 expected_entry_revision 必须为 null 或省略。' })
    }

    if (operation === 'modify_entry') {
        errors.push({ field: 'operation', message: 'Entry 不允许修改，只能使用 append_entry 追加 candidate Entry。' })
    }

    if (errors.length > 0) {
        return { ok: false, errors }
    }

    return { ok: true, value: normalizeUpdateTermEntriesInput(parsed.value) }
}

function hasTranslationTargetConstraint (target: EntryPayload['target']|GlossaryEntry['target']): boolean {
    if (!target) {
        return false
    }

    return Boolean(target.preferred_translation)
        || Boolean(target.alternative_translations?.length)
        || Boolean(target.forbidden_translations?.length)
}

function hasFlatTranslationTargetConstraint (value: Pick<AppendReviewEntryInput, 'preferred_translation'|'alternative_translations'|'forbidden_translations'>): boolean {
    return Boolean(value.preferred_translation)
        || Boolean(value.alternative_translations?.length)
        || Boolean(value.forbidden_translations?.length)
}

function validateEffectiveRevisedEntry (
    entry: GlossaryEntry,
    revisedEntry: ReviewEntryAction['revised_entry'],
    fieldPrefix: string,
): ValidationError[] {
    if (!revisedEntry) {
        return []
    }

    const nextEntryType = revisedEntry.entry_type ?? entry.entry_type
    const nextBasis = nextEntryType === 'translation_rule'
        ? hasOwn(revisedEntry as Record<string, unknown>, 'basis') ? revisedEntry.basis : entry.basis
        : hasOwn(revisedEntry as Record<string, unknown>, 'basis') ? revisedEntry.basis : undefined
    const nextTarget = nextEntryType === 'translation_rule'
        ? hasOwn(revisedEntry as Record<string, unknown>, 'target') ? revisedEntry.target : entry.target
        : hasOwn(revisedEntry as Record<string, unknown>, 'target') ? revisedEntry.target : undefined
    const errors: ValidationError[] = []

    if (nextEntryType === 'translation_rule') {
        if (!nextBasis) {
            errors.push({ field: `${fieldPrefix}.basis`, message: 'translation_rule 必须提供 basis。' })
        }
        if (!hasTranslationTargetConstraint(nextTarget)) {
            errors.push({ field: `${fieldPrefix}.target`, message: 'translation_rule 必须提供 target，且至少包含 preferred_translation、alternative_translations 或 forbidden_translations。' })
        }
        if (nextTarget && nextTarget.target_language !== DEFAULT_TARGET_LANGUAGE) {
            errors.push({ field: `${fieldPrefix}.target.target_language`, message: `target_language 必须为 ${DEFAULT_TARGET_LANGUAGE}。` })
        }
    } else {
        if (nextBasis) {
            errors.push({ field: `${fieldPrefix}.basis`, message: 'basis 只允许用于 translation_rule Entry。' })
        }
        if (nextTarget) {
            errors.push({ field: `${fieldPrefix}.target`, message: `${nextEntryType} Entry 不允许包含 target。` })
        }
    }

    return errors
}

function buildGlossaryEntryFromPayload (
    payload: EntryPayload|NonNullable<ReviewEntryAction['entry']>,
    metadata: {
        entry_id: string
        term_id: string
        status?: GlossaryEntry['status']
        evidence_ids: string[]
        created_by?: string
        created_from?: Record<string, unknown>
        revision: number
        created_at: string
        updated_at: string
    },
): GlossaryEntry {
    const base = {
        entry_id: metadata.entry_id,
        term_id: metadata.term_id,
        content: payload.content,
        applicability: payload.applicability,
        policy: payload.policy,
        status: metadata.status ?? ('status' in payload ? payload.status : 'candidate'),
        evidence_ids: metadata.evidence_ids,
        created_by: metadata.created_by,
        created_from: metadata.created_from,
        revision: metadata.revision,
        created_at: metadata.created_at,
        updated_at: metadata.updated_at,
    }

    if (payload.entry_type === 'translation_rule') {
        if (!payload.basis || !payload.target) {
            throw new Error('validated translation_rule payload missing basis or target')
        }

        return {
            ...base,
            entry_type: 'translation_rule',
            basis: payload.basis,
            target: payload.target,
        }
    }

    return {
        ...base,
        entry_type: payload.entry_type,
    }
}

function validateAppendTermEntriesBatchInput (input: unknown): { ok: true, value: AppendTermEntriesBatchInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<AppendTermEntriesBatchInput>(appendTermEntriesBatchSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    const errors: ValidationError[] = []
    const items = parsed.value.items.map((item, index) => {
        const { client_id: clientId, ...entryInput } = item
        const validation = validateUpdateTermEntriesInput(entryInput)

        if (!validation.ok) {
            errors.push(...validation.errors.map(error => ({
                ...error,
                field: `items[${index}].${error.field}`,
            })))
            return {
                ...normalizeUpdateTermEntriesInput(entryInput),
                client_id: clientId,
            }
        }

        return {
            ...validation.value,
            client_id: clientId,
        }
    })

    return errors.length > 0
        ? { ok: false, errors }
        : { ok: true, value: { items } }
}

function normalizeEntryForStorage (
    term: GlossaryTerm,
    entry: EntryPayload,
    fieldPrefix: string,
): { ok: true, entry: EntryPayload }|{ ok: false, errors: ValidationError[] } {
    const errors: ValidationError[] = []
    const applicability = entry.applicability

    if (entry.entry_type !== 'translation_rule') {
        if (applicability.source_variant_indexes || applicability.source_selectors) {
            errors.push({
                field: `${fieldPrefix}.applicability.source_variant_indexes`,
                message: '非 translation_rule Entry 不允许包含 source selector 字段。',
            })
        }

        const {
            source_selectors: _sourceSelectors,
            source_variant_indexes: _sourceVariantIndexes,
            ...cleanApplicability
        } = applicability

        return errors.length > 0
            ? { ok: false, errors }
            : {
                ok: true,
                entry: {
                    ...entry,
                    applicability: cleanApplicability,
                },
            }
    }

    if (applicability.source_selectors) {
        errors.push({
            field: `${fieldPrefix}.applicability.source_selectors`,
            message: '工具输入不接受内部 source_selectors，请使用 source_variant_indexes。',
        })
    }

    const resolvedSelectors = resolveSourceVariantIndexes(term, applicability.source_variant_indexes, `${fieldPrefix}.applicability.source_variant_indexes`)

    if (!resolvedSelectors.ok) {
        errors.push(...resolvedSelectors.errors)
    }

    if (errors.length > 0) {
        return { ok: false, errors }
    }

    const {
        source_selectors: _sourceSelectors,
        source_variant_indexes: _sourceVariantIndexes,
        ...cleanApplicability
    } = applicability

    return {
        ok: true,
        entry: {
            ...entry,
            applicability: {
                ...cleanApplicability,
                source_selectors: resolvedSelectors.ok ? resolvedSelectors.selectors : [],
            },
        },
    }
}

function validateStoredEntrySourceSelectors (term: GlossaryTerm, entry: EntryPayload|GlossaryEntry, fieldPrefix: string): ValidationError[] {
    const selectors = entry.applicability.source_selectors
    const hasAgentVariantIndexes = 'source_variant_indexes' in entry.applicability && entry.applicability.source_variant_indexes !== undefined

    if (entry.entry_type !== 'translation_rule') {
        return selectors || hasAgentVariantIndexes
            ? [{ field: `${fieldPrefix}.applicability.source_selectors`, message: '非 translation_rule Entry 不允许包含 source selector 字段。' }]
            : []
    }

    if (!selectors || selectors.length === 0) {
        return [{ field: `${fieldPrefix}.applicability.source_selectors`, message: 'translation_rule 必须包含内部 source_selectors。' }]
    }

    const variants = getSourceVariants(term)
    const errors: ValidationError[] = []

    selectors.forEach((selector, index) => {
        const variant = variants.find(item => item.variant_id === selector.variant_id)

        if (!variant) {
            errors.push({ field: `${fieldPrefix}.applicability.source_selectors[${index}]`, message: `source selector ${selector.variant_id} 不存在。` })
            return
        }

        if (variant.text !== selector.text) {
            errors.push({ field: `${fieldPrefix}.applicability.source_selectors[${index}]`, message: `source selector ${selector.variant_id} 文本快照不一致。` })
        }
    })

    return errors
}

function validateCreateTermMergeProposalInput (input: unknown): { ok: true, value: CreateTermMergeProposalInput }|{ ok: false, errors: ValidationError[] } {
    const parsed = parseToolInput<CreateTermMergeProposalInput>(looseCreateTermMergeProposalSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    if (parsed.value.source_term_id === parsed.value.target_term_id) {
        return {
            ok: false,
            errors: [{ field: 'target_term_id', message: 'source_term_id 和 target_term_id 不能相同。' }],
        }
    }

    return { ok: true, value: parsed.value }
}

function validateGlossaryPlanInput (input: unknown): {
    ok: true
    value: GlossaryPlanInput
}|{
    ok: false
    errors: ValidationError[]
} {
    const errors: ValidationError[] = []
    const parsed = parseToolInput<GlossaryPlanInput>(looseGlossaryPlanSchema, input)

    if (!parsed.ok) {
        return parsed
    }

    if (!isRecord(input)) {
        return {
            ok: false,
            errors: [
                {
                    field: '(root)',
                    message: 'submit_glossary_plan 参数必须是 JSON object。',
                },
            ],
        }
    }

    const { term_extraction_policy: policy } = parsed.value

    validateExactSet('term_extraction_policy.entry_types', policy.entry_types, entryTypeValues, errors)

    if (errors.length > 0) {
        return {
            ok: false,
            errors,
        }
    }

    return {
        ok: true,
        value: parsed.value,
    }
}

function entryMatchesFilter (entry: GlossaryEntry, filter: EntryFilter|null): boolean {
    if (!filter) {
        return true
    }

    if (filter.entry_types && !filter.entry_types.includes(entry.entry_type)) {
        return false
    }

    if (filter.entry_statuses && !filter.entry_statuses.includes(entry.status)) {
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
        const fields = filter.text_fields ?? ['content.summary', 'content.description']
        const matcher = createTextMatcher(filter.text, filter.is_regex, filter.case_sensitive)

        if (!fields.some(field => valuesForEntryField(entry, field).some(value => matcher(value)))) {
            return false
        }
    }

    return true
}

function termMatchesFilter (term: GlossaryTerm|undefined, filter: TermFilter|null): boolean {
    if (!filter) {
        return true
    }

    if (!term) {
        return false
    }

    if (filter.source_language && term.source_language !== filter.source_language) {
        return false
    }

    if (filter.term_types && !filter.term_types.includes(term.term_type)) {
        return false
    }

    if (filter.term_statuses && !filter.term_statuses.includes(term.status)) {
        return false
    }

    return true
}

function createTermMatcher (queryText: string, mode: MatchMode, caseSensitive: boolean): (term: GlossaryTerm) => string|null {
    if (mode === 'regex') {
        const regex = new RegExp(queryText, caseSensitive ? '' : 'i')
        return term => {
            const candidates = [term.source_text, ...getAliasTexts(term)]
            const matched = candidates.find(candidate => regex.test(candidate))
            regex.lastIndex = 0
            return matched ?? null
        }
    }

    const normalizedQuery = caseSensitive ? normalizeWhitespace(queryText) : normalizeSourceText(queryText)

    if (mode === 'exact') {
        return term => compareText(term.source_text, normalizedQuery, caseSensitive) ? term.source_text : null
    }

    if (mode === 'alias') {
        return term => getAliasTexts(term).find(alias => compareText(alias, normalizedQuery, caseSensitive)) ?? null
    }

    if (mode === 'compound') {
        return term => {
            const source = caseSensitive ? normalizeWhitespace(term.source_text) : normalizeSourceText(term.source_text)
            const aliases = getAliasTexts(term).map(alias => caseSensitive ? normalizeWhitespace(alias) : normalizeSourceText(alias))

            return normalizedQuery.includes(source) || source.includes(normalizedQuery) || aliases.some(alias => normalizedQuery.includes(alias) || alias.includes(normalizedQuery))
                ? term.source_text
                : null
        }
    }

    return term => {
        const candidates = [term.source_text, ...getAliasTexts(term)]
        const matched = candidates.find(candidate => similarity(caseSensitive ? normalizeWhitespace(candidate) : normalizeSourceText(candidate), normalizedQuery) >= 0.72)
        return matched ?? null
    }
}

function compareText (value: string, normalizedQuery: string, caseSensitive: boolean): boolean {
    return (caseSensitive ? normalizeWhitespace(value) : normalizeSourceText(value)) === normalizedQuery
}

function entryStatusAllowed (entry: GlossaryEntry, statuses: EntryStatus[]|null): boolean {
    return !statuses || statuses.includes(entry.status)
}

function evidenceEntryAllowed (
    evidence: GlossaryEvidence,
    includedEntries: GlossaryEntry[],
    includeEntries: boolean,
    statuses: EntryStatus[]|null,
    state: GlossaryState,
): boolean {
    if (includeEntries) {
        return includedEntries.some(entry => entry.entry_id === evidence.entry_id)
    }

    if (!statuses) {
        return true
    }

    const entry = state.entries.find(candidate => candidate.entry_id === evidence.entry_id)
    return !!entry && statuses.includes(entry.status)
}

function allowsEvidenceOptionalTranslationRule (entry: Pick<EntryPayload|GlossaryEntry, 'entry_type'> & { basis?: EntryPayload['basis']|GlossaryEntry['basis']|null }): boolean {
    return entry.entry_type === 'translation_rule'
        && (entry.basis === 'transliteration' || entry.basis === 'project_convention')
}

async function normalizeEvidencePayloads (
    root: string,
    evidence: EvidencePayload[],
    manualTrans: ManualTransData,
    evidenceScope: EvidenceScope|undefined,
    allowedEvidenceRanges: EvidenceRange[],
): Promise<{ evidence: NormalizedEvidencePayload[], errors: ValidationError[] }> {
    const errors: ValidationError[] = []
    const normalizedEvidence: NormalizedEvidencePayload[] = []
    if (!evidenceScope && allowedEvidenceRanges.length === 0) {
        return {
            evidence: [],
            errors: evidence.map((_item, index) => ({
                field: `evidence[${index}].source_ref.filtered_key_index`,
                message: 'Evidence 必须引用当前 batch 或 get_evidence_context 返回范围内的源文本。',
            })),
        }
    }

    const rawIndexByFilteredIndex = new Map<number, number>(
        evidenceScope?.filtered_index_to_key_index
            ? Object.entries(evidenceScope.filtered_index_to_key_index).map(([filteredIndex, keyIndex]) => [Number(filteredIndex), keyIndex])
            : Array.from((await createFilteredIndexByRawIndex(root, manualTrans)).entries()).map(([rawIndex, filteredIndex]) => [filteredIndex, rawIndex]),
    )

    evidence.forEach((item, index) => {
        const sourceFileId = item.source_ref.source_file_id
        const filteredKeyIndex = item.source_ref.filtered_key_index
        const keyIndex = rawIndexByFilteredIndex.get(filteredKeyIndex)
        const keyText = typeof keyIndex === 'number' && keyIndex >= 0 && keyIndex < manualTrans.keys.length
            ? manualTrans.keys[keyIndex]
            : null

        if (!item.source_ref.key_hash && normalizeProjectFileId(sourceFileId) !== normalizeProjectFileId(manualTrans.relativePath)) {
            errors.push({
                field: `evidence[${index}].source_ref.source_file_id`,
                message: `Evidence source_file_id 必须指向当前配置源文件 ${manualTrans.relativePath}。`,
            })
        }

        if (typeof keyIndex !== 'number' || keyText === null) {
            errors.push({
                field: `evidence[${index}].source_ref.filtered_key_index`,
                message: 'Evidence 必须提供可定位的 filtered_key_index。',
            })
            return
        }

        if (item.source_ref.key_hash && normalizeKeyHash(item.source_ref.key_hash) !== sha256Text(keyText)) {
            errors.push({
                field: `evidence[${index}].source_ref.key_hash`,
                message: 'key_hash 与 filtered_key_index 定位的源文本不一致。',
            })
        }

        if (!keyText.includes(item.quote)) {
            errors.push({
                field: `evidence[${index}].quote`,
                message: 'quote 必须出现在 filtered_key_index 定位的源文本中。',
            })
        }

        if (!isEvidenceKeyIndexAllowed(sourceFileId, keyIndex, filteredKeyIndex, evidenceScope, allowedEvidenceRanges)) {
            errors.push({
                field: `evidence[${index}].source_ref.filtered_key_index`,
                message: 'Evidence 必须引用当前 batch 或 get_evidence_context 返回范围内的源文本。',
            })
        }

        normalizedEvidence.push({
            quote: item.quote,
            context: item.context,
            reason: item.reason,
            source_ref: {
                source_file_id: item.source_ref.source_file_id,
                key_index: keyIndex,
                key_hash: item.source_ref.key_hash,
                span: item.source_ref.span,
            },
        })
    })

    return { evidence: errors.length === 0 ? normalizedEvidence : [], errors }
}

function noQuerySnapshotOutput (message: string, termId?: string): unknown {
    return {
        ok: false,
        code: 'no_query_snapshot',
        message,
        ...(termId ? { term_id: termId } : {}),
        retry_required: true,
        retry_tool: 'query_glossary_terms',
        retry_query: {},
    }
}

// Differs from translation tool validation: not retryable unless a validator opts in.
function validationErrorOutput (errors: ValidationError[]): unknown {
    return {
        ok: false,
        code: 'validation_error',
        errors,
        retry_required: false,
    }
}

function invalidAliasOutput (errors: ValidationError[]): unknown {
    return {
        ok: false,
        code: 'invalid_alias',
        errors,
        retry_required: false,
    }
}

function validationOutput (errors: ValidationError[]): unknown {
    return errors.some(error => error.code === 'invalid_alias')
        ? invalidAliasOutput(errors)
        : validationErrorOutput(errors)
}

function decorateBatchResult (result: unknown, index: number, clientId: string): Record<string, unknown> {
    return {
        index,
        client_id: clientId,
        ...(isRecord(result) ? result : { ok: false, code: 'invalid_result', result }),
    }
}

function batchOutput (results: Record<string, unknown>[]): unknown {
    const failed = results.filter(result => result.ok !== true)

    return {
        ok: failed.length < results.length,
        partial_success: failed.length > 0 && failed.length < results.length,
        results,
        retry_required: results.some(result => result.retry_required === true),
    }
}

function evidenceContextValidationCode (errors: ValidationError[]): 'invalid_evidence_id'|'window_too_large' {
    return errors.some(error => error.field === 'window_before' || error.field === 'window_after' || error.field === 'max_chars_per_key')
        ? 'window_too_large'
        : 'invalid_evidence_id'
}

function parseToolInput<T> (schema: ZodTypeAny, input: unknown): { ok: true, value: T }|{ ok: false, errors: ValidationError[] } {
    const parsed = schema.safeParse(input)

    if (!parsed.success) {
        return {
            ok: false,
            errors: parsed.error.issues.map(issue => ({
                field: formatZodIssuePath(issue.path),
                message: issue.message,
            })),
        }
    }

    return {
        ok: true,
        value: parsed.data as T,
    }
}

function formatZodIssuePath (path: (string|number)[]): string {
    if (path.length === 0) {
        return '(root)'
    }

    return path.reduce<string>((formatted, item) => {
        if (typeof item === 'number') {
            return `${formatted}[${item}]`
        }

        return formatted ? `${formatted}.${item}` : item
    }, '')
}

function runMutableOrCache (
    state: GlossaryState,
    reviewMode: boolean|undefined,
    toolName: string,
    input: Record<string, unknown>,
    run: () => unknown,
): unknown {
    // Dead code unless worker/review parallelism returns.
    if (state.review.frozen && !reviewMode) {
        const now = new Date().toISOString()
        const cacheId = createNextCacheId(state)

        state.review.cached_worker_writes.push({
            cache_id: cacheId,
            tool_name: toolName,
            input: structuredClone(input) as Record<string, unknown>,
            status: 'pending',
            created_at: now,
        })
        state.meta.updated_at = now

        return {
            ok: true,
            action: 'cached',
            cache_id: cacheId,
            message: '术语表正在审核冻结中，本次 worker 写入已进入 cache layer，审核完成后由程序侧重新校验并落入。',
            retry_required: false,
        }
    }

    return run()
}

async function collectReviewStructuralIssues (
    root: string,
    manualTransFile: string,
    state: GlossaryState,
    window: GlossaryReviewWindow,
    terms: GlossaryTerm[],
    entries: GlossaryEntry[],
    evidence: GlossaryEvidence[],
): Promise<unknown[]> {
    const issues: unknown[] = []

    for (const entry of entries) {
        if (entry.status === 'candidate' && entry.evidence_ids.length === 0 && !allowsEvidenceOptionalTranslationRule(entry)) {
            issues.push({
                kind: 'entry_without_evidence',
                severity: 'error',
                entry_id: entry.entry_id,
                term_id: entry.term_id,
            })
        }
    }

    const effectiveEntries = collectEffectiveEntryStates(state, window.pending_entry_actions)
    for (const term of terms) {
        if (effectiveTermStatus(term, window) !== 'active') {
            continue
        }

        const hasEffectiveEntry = effectiveEntries.some(entry => (
            entry.term_id === term.term_id
            && (entry.status === 'approved' || entry.status === 'candidate')
        ))

        if (!hasEffectiveEntry) {
            issues.push({
                kind: 'active_term_without_effective_entries',
                severity: 'warning',
                term_id: term.term_id,
                source_text: term.source_text,
                message: 'Active term has no approved or candidate entries; reject/deprecate it, add a justified entry, or explain defer in review summary.',
            })
        }
    }

    if (evidence.length === 0) {
        return issues
    }

    const manualTrans = await loadManualTransData(root, manualTransFile).catch(() => null)
    if (!manualTrans) {
        return issues
    }

    for (const item of evidence) {
        const keyIndex = item.source_ref.key_index
        const keyText = typeof keyIndex === 'number' ? manualTrans.keys[keyIndex] : undefined

        if (keyIndex === undefined || !keyText || !keyText.includes(item.quote)) {
            issues.push({
                kind: 'quote_invalid',
                severity: 'warning',
                evidence_id: item.evidence_id,
                entry_id: item.entry_id,
                term_id: item.term_id,
            })
        }
    }

    return issues
}

function getPendingReviewBatchNumbers (state: GlossaryState, batchNumbers?: number[]): number[] {
    const allowedBatchNumbers = batchNumbers ? new Set(batchNumbers) : null

    return state.review.completed_batches
        .filter(batchNumber => batchNumber > state.review.last_reviewed_batch_number)
        .filter(batchNumber => !allowedBatchNumbers || allowedBatchNumbers.has(batchNumber))
        .sort((left, right) => left - right)
}

function getActiveReviewWindow (state: GlossaryState, reviewWindowId: string): GlossaryReviewWindow {
    const window = state.review.active_window

    if (!window || window.review_window_id !== reviewWindowId) {
        throw new Error(`No active review window ${reviewWindowId}.`)
    }

    return window
}

function applyReviewEntryActions (state: GlossaryState, actions: ReviewEntryAction[]): void {
    for (const action of actions) {
        if (action.operation === 'append') {
            applyAppendReviewEntryAction(state, action)
            continue
        }

        const entry = state.entries.find(item => item.entry_id === action.entry_id)

        if (!entry) {
            addReviewConflict(state, 'review_entries_batch', `Entry ${action.entry_id} 不存在。`, { action })
            continue
        }

        if (action.expected_entry_revision && action.expected_entry_revision !== entry.revision) {
            addReviewConflict(state, 'review_entries_batch', `Entry ${action.entry_id} revision 不一致。`, { action })
            continue
        }

        const now = new Date().toISOString()
        const entryTerm = state.terms.find(item => item.term_id === entry.term_id)

        if (action.operation === 'approve') {
            if (entryTerm) {
                const storedValidation = validateStoredEntrySourceSelectors(entryTerm, entry, 'entry')
                if (storedValidation.length > 0) {
                    addReviewConflict(state, 'review_entries_batch', `Entry ${entry.entry_id} source selector 无效。`, { action, errors: storedValidation })
                    continue
                }
            }
            entry.status = 'approved'
        } else if (action.operation === 'reject') {
            entry.status = 'rejected'
            removeGenderPresentationForEntry(state, entry.entry_id)
            entry.policy.notes = uniqueStrings([...(entry.policy.notes ?? []), `review rejected: ${action.reason}`])
        } else if (action.operation === 'revise') {
            const revised = applyRevisedEntryFields(entry, action.revised_entry, entryTerm, 'revised_entry')
            if (!revised.ok) {
                addReviewConflict(state, 'review_entries_batch', `Entry ${entry.entry_id} revised source selector 无效。`, { action, errors: revised.errors })
                continue
            }
            entry.status = 'approved'
        } else if (action.operation === 'merge_into') {
            const targetEntry = action.target_entry_id
                ? state.entries.find(item => item.entry_id === action.target_entry_id)
                : undefined

            if (!targetEntry) {
                addReviewConflict(state, 'review_entries_batch', 'merge_into 缺少有效 target_entry_id。', { action })
                continue
            }

            transferGenderPresentationBetweenEntries(state, entry, targetEntry)
            targetEntry.evidence_ids = uniqueStrings([...targetEntry.evidence_ids, ...entry.evidence_ids])
            for (const evidence of state.evidence.filter(item => item.entry_id === entry.entry_id)) {
                evidence.entry_id = targetEntry.entry_id
                evidence.term_id = targetEntry.term_id
            }
            entry.evidence_ids = []
            entry.status = 'rejected'
            targetEntry.revision += 1
            targetEntry.updated_at = now
        } else if (action.operation === 'move_to_term' || action.operation === 'move_to_term_and_approve') {
            if (!action.target_term_id) {
                addReviewConflict(state, 'review_entries_batch', `${action.operation} 缺少 target_term_id。`, { action })
                continue
            }
            const targetTerm = state.terms.find(item => item.term_id === action.target_term_id)
            if (!targetTerm) {
                addReviewConflict(state, 'review_entries_batch', `target Term ${action.target_term_id} 不存在。`, { action })
                continue
            }
            const hasRevisedSourceVariantIndexes = Array.isArray(action.revised_entry?.applicability?.source_variant_indexes)
            if (action.revised_entry) {
                const sourceTerm = state.terms.find(item => item.term_id === entry.term_id)
                const targetTermPreview = structuredClone(targetTerm) as GlossaryTerm
                const entryPreview = structuredClone(entry) as GlossaryEntry
                if (!hasRevisedSourceVariantIndexes && sourceTerm) {
                    remapEntrySourceSelectorsForMove(entryPreview, sourceTerm, targetTermPreview)
                }
                const revised = applyRevisedEntryFields(entryPreview, withFallbackSourceVariantIndexes(entryPreview, action.revised_entry, targetTermPreview), targetTermPreview, 'revised_entry')
                if (!revised.ok) {
                    addReviewConflict(state, 'review_entries_batch', `Entry ${entry.entry_id} revised source selector 无效。`, { action, errors: revised.errors })
                    continue
                }
            }

            moveEntryToTerm(state, entry, action.target_term_id, !hasRevisedSourceVariantIndexes)

            if (action.revised_entry) {
                const revised = applyRevisedEntryFields(entry, withFallbackSourceVariantIndexes(entry, action.revised_entry, targetTerm), targetTerm, 'revised_entry')
                if (!revised.ok) {
                    addReviewConflict(state, 'review_entries_batch', `Entry ${entry.entry_id} revised source selector 无效。`, { action, errors: revised.errors })
                    continue
                }
            }

            if (action.operation === 'move_to_term_and_approve') {
                entry.status = 'approved'
            }
        }

        if (entry.entry_type !== 'translation_rule') {
            delete entry.target
            delete entry.basis
            delete entry.applicability.source_selectors
        } else {
            removeGenderPresentationForEntry(state, entry.entry_id)
            const currentEntryTerm = state.terms.find(item => item.term_id === entry.term_id)
            if (currentEntryTerm) {
                const storedValidation = validateStoredEntrySourceSelectors(currentEntryTerm, entry, 'entry')
                if (storedValidation.length > 0) {
                    addReviewConflict(state, 'review_entries_batch', `Entry ${entry.entry_id} source selector 无效。`, { action, errors: storedValidation })
                    continue
                }
            }
        }

        entry.revision += 1
        entry.updated_at = now
    }
}

function applyAppendReviewEntryAction (state: GlossaryState, action: ReviewEntryAction): void {
    const termId = action.term_id
    const term = termId ? state.terms.find(item => item.term_id === termId) : undefined

    if (!term) {
        addReviewConflict(state, 'append_review_entry', 'append 缺少有效 Term。', { action })
        return
    }

    if (term.status !== 'active') {
        addReviewConflict(state, 'append_review_entry', `Term ${term.term_id} 不是 active，不能追加 Entry。`, { action })
        return
    }

    if (!action.entry) {
        addReviewConflict(state, 'append_review_entry', 'append 缺少 Entry 内容。', { action })
        return
    }

    const copiedEvidence = (action.evidence_ids ?? []).map(evidenceId => state.evidence.find(item => item.evidence_id === evidenceId))

    if (copiedEvidence.some(item => !item || item.term_id !== term.term_id)) {
        addReviewConflict(state, 'append_review_entry', 'append 引用的 evidence_ids 不存在或不属于目标 Term。', { action })
        return
    }

    const now = new Date().toISOString()
    const entryId = createNextEntryId(state, term.term_id)
    const evidenceCount = copiedEvidence.length + (action.evidence?.length ?? 0)
    const evidenceIds = createNextEvidenceIds(state, term.term_id, evidenceCount)
    const entry = buildGlossaryEntryFromPayload(action.entry, {
        entry_id: entryId,
        term_id: term.term_id,
        status: 'approved',
        evidence_ids: evidenceIds,
        created_by: 'review-agent',
        created_from: {
            review_window_id: state.review.active_window?.review_window_id,
            tool: 'append_review_entry',
            action_id: action.action_id,
        },
        revision: 1,
        created_at: now,
        updated_at: now,
    })

    if (entry.entry_type !== 'translation_rule') {
        delete entry.target
        delete entry.basis
        delete entry.applicability.source_selectors
    } else {
        const storedValidation = validateStoredEntrySourceSelectors(term, entry, 'entry')
        if (storedValidation.length > 0) {
            addReviewConflict(state, 'append_review_entry', 'Entry source selector 无效。', { action, errors: storedValidation })
            return
        }
    }

    const copiedEvidenceRecords: GlossaryEvidence[] = copiedEvidence
        .filter((item): item is GlossaryEvidence => !!item)
        .map((item, index) => ({
            evidence_id: evidenceIds[index],
            term_id: term.term_id,
            entry_id: entryId,
            source_ref: {
                source_file_id: item.source_ref.source_file_id,
                file_id: item.source_ref.file_id,
                key_index: item.source_ref.key_index,
                key_hash: item.source_ref.key_hash,
                span: item.source_ref.span,
            },
            quote: item.quote,
            context: item.context,
            reason: item.reason,
            created_by: 'review-agent',
            created_at: now,
        }))
    const newEvidenceOffset = copiedEvidenceRecords.length
    const newEvidenceRecords: GlossaryEvidence[] = (action.evidence ?? []).map((item, index) => ({
        evidence_id: evidenceIds[newEvidenceOffset + index],
        term_id: term.term_id,
        entry_id: entryId,
        source_ref: {
            source_file_id: item.source_ref.source_file_id,
            file_id: item.source_ref.file_id,
            key_index: item.source_ref.key_index,
            key_hash: item.source_ref.key_hash ? normalizeKeyHashForStorage(item.source_ref.key_hash) : undefined,
            span: item.source_ref.span,
        },
        quote: item.quote,
        context: item.context,
        reason: item.reason,
        created_by: 'review-agent',
        created_at: now,
    }))

    state.entries.push(entry)
    state.evidence.push(...copiedEvidenceRecords, ...newEvidenceRecords)
    term.entry_ids = uniqueStrings([...term.entry_ids, entryId])
    term.revision += 1
    term.updated_at = now
}

function withFallbackSourceVariantIndexes (
    entry: GlossaryEntry,
    revisedEntry: NonNullable<ReviewEntryAction['revised_entry']>,
    term: GlossaryTerm,
): NonNullable<ReviewEntryAction['revised_entry']> {
    if (!revisedEntry.applicability || revisedEntry.applicability.source_variant_indexes) {
        return revisedEntry
    }

    const nextEntryType = revisedEntry.entry_type ?? entry.entry_type
    if (nextEntryType !== 'translation_rule') {
        return revisedEntry
    }

    const fallbackIndexes = sourceSelectorsToVariantIndexes(term, entry.applicability.source_selectors)
    if (fallbackIndexes.length === 0) {
        return revisedEntry
    }

    return {
        ...revisedEntry,
        applicability: {
            ...revisedEntry.applicability,
            source_variant_indexes: fallbackIndexes,
        },
    }
}

function sourceSelectorsToVariantIndexes (term: GlossaryTerm, selectors: GlossaryEntry['applicability']['source_selectors']): number[] {
    if (!selectors || selectors.length === 0) {
        return []
    }

    const variants = getSourceVariants(term)
    return selectors
        .map(selector => variants.find(variant => variant.variant_id === selector.variant_id && variant.text === selector.text)?.index)
        .filter((index): index is number => typeof index === 'number')
}

function applyRevisedEntryFields (
    entry: GlossaryEntry,
    revisedEntry: ReviewEntryAction['revised_entry'],
    term: GlossaryTerm|undefined,
    fieldPrefix: string,
): { ok: true }|{ ok: false, errors: ValidationError[] } {
    if (!revisedEntry) {
        return { ok: true }
    }

    const preview = buildRevisedEntryPreview(entry, revisedEntry, term, fieldPrefix)
    if (!preview.ok) {
        return preview
    }

    replaceEntryFields(entry, preview.entry)
    return { ok: true }
}

function buildRevisedEntryPreview (
    entry: GlossaryEntry,
    revisedEntry: NonNullable<ReviewEntryAction['revised_entry']>,
    term: GlossaryTerm|undefined,
    fieldPrefix: string,
): { ok: true, entry: GlossaryEntry }|{ ok: false, errors: ValidationError[] } {
    const effectiveEntryErrors = validateEffectiveRevisedEntry(entry, revisedEntry, fieldPrefix)
    if (effectiveEntryErrors.length > 0) {
        return { ok: false, errors: effectiveEntryErrors }
    }

    const preview = structuredClone(entry) as GlossaryEntry
    const nextEntryType = revisedEntry.entry_type ?? entry.entry_type
    const nextBasis = hasOwn(revisedEntry as Record<string, unknown>, 'basis') ? revisedEntry.basis : entry.basis
    const nextContent = revisedEntry.content ?? entry.content
    const nextTarget = hasOwn(revisedEntry as Record<string, unknown>, 'target') ? revisedEntry.target : entry.target
    const nextPolicy = revisedEntry.policy ?? entry.policy
    let nextApplicability = entry.applicability

    if (revisedEntry.applicability) {
        if (!term) {
            return {
                ok: false,
                errors: [{
                    field: `${fieldPrefix}.applicability`,
                    message: 'revised_entry.applicability 需要对应的 Term 上下文。',
                }],
            }
        }

        const normalizedEntry = normalizeEntryForStorage(term, {
            entry_type: nextEntryType,
            basis: nextBasis,
            content: nextContent,
            target: nextTarget,
            applicability: revisedEntry.applicability,
            policy: nextPolicy,
            status: entry.status,
        }, fieldPrefix)

        if (!normalizedEntry.ok) {
            return normalizedEntry
        }

        nextApplicability = normalizedEntry.entry.applicability
    }

    preview.entry_type = nextEntryType
    if (nextBasis) {
        preview.basis = nextBasis
    } else {
        delete preview.basis
    }
    preview.content = nextContent
    if (nextTarget) {
        preview.target = nextTarget
    } else {
        delete preview.target
    }
    preview.applicability = nextApplicability
    preview.policy = nextPolicy

    if (preview.entry_type !== 'translation_rule') {
        delete preview.target
        delete preview.basis
        delete preview.applicability.source_selectors
    }

    if (preview.entry_type === 'translation_rule') {
        if (!term) {
            return {
                ok: false,
                errors: [{
                    field: `${fieldPrefix}.applicability`,
                    message: 'translation_rule 需要对应的 Term 上下文以校验 source selector。',
                }],
            }
        }

        const storedValidation = validateStoredEntrySourceSelectors(term, preview, fieldPrefix)
        if (storedValidation.length > 0) {
            return {
                ok: false,
                errors: storedValidation.map(error => formatRevisedEntrySourceSelectorError(error, fieldPrefix)),
            }
        }
    }

    return { ok: true, entry: preview }
}

function formatRevisedEntrySourceSelectorError (error: ValidationError, fieldPrefix: string): ValidationError {
    const sourceSelectorsPath = `${fieldPrefix}.applicability.source_selectors`
    if (!error.field.startsWith(sourceSelectorsPath)) {
        return error
    }

    return {
        field: error.field.replace(sourceSelectorsPath, `${fieldPrefix}.applicability.source_variant_indexes`),
        message: error.message === 'translation_rule 必须包含内部 source_selectors。'
            ? 'translation_rule 必须提供 applicability.source_variant_indexes。'
            : error.message,
    }
}

function replaceEntryFields (entry: GlossaryEntry, replacement: GlossaryEntry): void {
    for (const key of Object.keys(entry) as Array<keyof GlossaryEntry>) {
        delete entry[key]
    }

    Object.assign(entry, replacement)
}

function applyReviewTermActions (state: GlossaryState, actions: ReviewTermAction[]): void {
    for (const action of actions) {
        const term = state.terms.find(item => item.term_id === action.term_id)

        if (!term) {
            addReviewConflict(state, 'review_terms_batch', `Term ${action.term_id} 不存在。`, { action })
            continue
        }

        if (action.expected_term_revision && action.expected_term_revision !== term.revision) {
            addReviewConflict(state, 'review_terms_batch', `Term ${action.term_id} revision 不一致。`, { action })
            continue
        }

        const now = new Date().toISOString()

        if (action.operation === 'reject') {
            rejectTerm(term, action.rejected_reason)
            for (const entry of state.entries.filter(item => item.term_id === term.term_id)) {
                if (entry.status === 'candidate') {
                    entry.status = 'rejected'
                    entry.revision += 1
                    entry.updated_at = now
                } else if (entry.status === 'approved') {
                    addReviewConflict(state, 'review_terms_batch', `Term ${term.term_id} 被拒绝时 Entry ${entry.entry_id} 已是 approved。`, { action, entry_id: entry.entry_id })
                    entry.status = 'rejected'
                    entry.policy.notes = uniqueStrings([...(entry.policy.notes ?? []), `review rejected with term: ${action.reason}`])
                    entry.revision += 1
                    entry.updated_at = now
                }
            }
        } else if (action.operation === 'deprecate') {
            term.status = 'deprecated'
            clearRejectedReason(term)
        } else if (action.operation === 'change_term_type') {
            if (!action.term_type) {
                addReviewConflict(state, 'review_terms_batch', 'change_term_type 缺少 term_type。', { action })
                continue
            }
            term.term_type = action.term_type
            normalizeTermGenderPresentations(state, term)
        } else if (action.operation === 'add_aliases') {
            const aliasErrors: ValidationError[] = []
            validateAliases('aliases_to_add', action.aliases_to_add ?? [], term.source_language, aliasErrors)
            if (aliasErrors.length > 0) {
                addReviewConflict(state, 'review_terms_batch', aliasErrors.map(error => error.message).join('；'), { action })
                continue
            }

            addAliasesToTerm(term, action.aliases_to_add ?? [])
        } else if (action.operation === 'remove_invalid_aliases') {
            const referencedAliasIds = findReferencedAliasIds(state, term, action.aliases_to_remove ?? [])
            if (referencedAliasIds.length > 0) {
                addReviewConflict(state, 'review_terms_batch', `remove_invalid_aliases 引用中: ${referencedAliasIds.join(', ')}。`, { action })
                continue
            }
            removeAliasesFromTermByText(term, action.aliases_to_remove ?? [])
        } else if (action.operation === 'merge_term') {
            if (!action.target_term_id) {
                addReviewConflict(state, 'review_terms_batch', 'merge_term 缺少 target_term_id。', { action })
                continue
            }
            mergeTermIntoTarget(state, term, action.target_term_id)
        } else if (action.operation === 'move_entries') {
            if (!action.target_term_id) {
                addReviewConflict(state, 'review_terms_batch', 'move_entries 缺少 target_term_id。', { action })
                continue
            }
            for (const entryId of action.entry_ids ?? []) {
                const entry = state.entries.find(item => item.entry_id === entryId)
                if (entry) {
                    moveEntryToTerm(state, entry, action.target_term_id)
                }
            }
        } else if (action.operation === 'set_gender_presentations') {
            const presentations = action.gender_presentations ?? []
            const genderErrors = validateGenderPresentationsForTerm(state, term, presentations, 'gender_presentations')
            if (genderErrors.length > 0) {
                addReviewConflict(state, 'review_terms_batch', `Term ${term.term_id} gender_presentations 无效。`, { action, errors: genderErrors })
                continue
            }
            term.gender_presentations = presentations
            normalizeTermGenderPresentations(state, term)
        }

        term.revision += 1
        term.updated_at = now
    }
}

function rejectTerm (term: GlossaryTerm, reason: RejectedReason|undefined): void {
    term.status = 'rejected'
    term.rejected_reason = reason ?? defaultRejectedReasonForTerm(term)
}

function clearRejectedReason (term: GlossaryTerm): void {
    delete term.rejected_reason
}

function defaultRejectedReasonForTerm (term: GlossaryTerm): RejectedReason {
    return term.entry_ids.length === 0
        ? 'empty_insufficient_evidence'
        : 'low_translation_value'
}

async function replayCachedWorkerWrites (root: string, manualTransFile: string, state: GlossaryState): Promise<{ applied: number, conflicted: number }> {
    // Dead code unless worker/review parallelism returns.
    let applied = 0
    let conflicted = 0

    for (const write of state.review.cached_worker_writes.filter(item => item.status === 'pending')) {
        const result = await applyCachedWorkerWrite(root, manualTransFile, state, write.tool_name, write.input)

        if (result.ok) {
            write.status = 'applied'
            write.applied_at = new Date().toISOString()
            applied += 1
        } else {
            write.status = 'conflict'
            write.conflict_reason = result.reason
            addReviewConflict(state, write.tool_name, write.conflict_reason, write.input, write.cache_id)
            conflicted += 1
        }
    }

    return { applied, conflicted }
}

async function applyCachedWorkerWrite (
    root: string,
    manualTransFile: string,
    state: GlossaryState,
    toolName: string,
    input: Record<string, unknown>,
): Promise<{ ok: true }|{ ok: false, reason: string }> {
    if (toolName === 'create_or_get_term') {
        const validation = validateCreateOrGetTermInput(input)
        return validation.ok ? applyCachedCreateOrGetTerm(state, validation.value) : cacheValidationFailure(validation.errors)
    }

    if (toolName === 'create_or_get_terms') {
        const validation = validateCreateOrGetTermsInput(input)
        if (!validation.ok) {
            return cacheValidationFailure(validation.errors)
        }

        for (const item of validation.value.items) {
            const { client_id: _clientId, ...termInput } = item
            const result = applyCachedCreateOrGetTerm(state, termInput)
            if (!result.ok) {
                return result
            }
        }
        return { ok: true }
    }

    if (toolName === 'update_term_metadata') {
        const validation = validateUpdateTermMetadataInput(input)
        return validation.ok ? applyCachedUpdateTermMetadata(state, validation.value) : cacheValidationFailure(validation.errors)
    }

    if (toolName === 'update_term_entries') {
        const validation = validateUpdateTermEntriesInput(input)
        return validation.ok ? applyCachedUpdateTermEntries(root, manualTransFile, state, validation.value) : cacheValidationFailure(validation.errors)
    }

    if (toolName === 'append_term_entries_batch') {
        const validation = validateAppendTermEntriesBatchInput(input)
        if (!validation.ok) {
            return cacheValidationFailure(validation.errors)
        }

        for (const item of validation.value.items) {
            const { client_id: _clientId, ...entryInput } = item
            const result = await applyCachedUpdateTermEntries(root, manualTransFile, state, entryInput)
            if (!result.ok) {
                return result
            }
        }
        return { ok: true }
    }

    if (toolName === 'create_term_merge_proposal') {
        const validation = validateCreateTermMergeProposalInput(input)
        return validation.ok ? applyCachedMergeProposal(root, manualTransFile, state, validation.value) : cacheValidationFailure(validation.errors)
    }

    return {
        ok: false,
        reason: `unsupported cached worker write tool: ${toolName}`,
    }
}

function applyCachedCreateOrGetTerm (state: GlossaryState, input: CreateOrGetTermInput): { ok: true }|{ ok: false, reason: string } {
    const exactDuplicateTerm = findExactDuplicateTerm(state, input)

    if (exactDuplicateTerm) {
        return exactDuplicateTerm.status === 'merged'
            ? { ok: false, reason: `cached create_or_get_term matched merged term ${exactDuplicateTerm.term_id}` }
            : { ok: true }
    }

    const possibleDuplicateTerms = findPossibleDuplicateTerms(state, input)
        .filter(term => !(input.confirmed_distinct_from_term_ids ?? []).includes(term.term_id))

    if (possibleDuplicateTerms.length > 0) {
        return {
            ok: false,
            reason: `cached create_or_get_term has possible duplicate terms: ${possibleDuplicateTerms.map(term => term.term_id).join(', ')}`,
        }
    }

    const now = new Date().toISOString()
    const termId = createNextTermId(state)
    const aliasState = createAliasState(termId, input.aliases)
    state.terms.push({
        term_id: termId,
        source_text: input.source_text,
        source_language: input.source_language,
        term_type: input.term_type,
        ...aliasState,
        status: 'active',
        merged_into: null,
        entry_ids: [],
        created_at: now,
        updated_at: now,
        revision: 1,
        created_by: input.created_by,
        created_from: input.created_from,
    })

    return { ok: true }
}

function applyCachedUpdateTermMetadata (state: GlossaryState, input: UpdateTermMetadataInput): { ok: true }|{ ok: false, reason: string } {
    const term = resolveCacheTargetTerm(state, input.term_id)

    if (!term.ok) {
        return term
    }

    if (term.term.revision !== input.expected_term_revision) {
        return {
            ok: false,
            reason: `cached update_term_metadata revision conflict for ${input.term_id}`,
        }
    }

    const now = new Date().toISOString()

    if (input.operation === 'add_aliases') {
        const aliasErrors: ValidationError[] = []
        validateAliases('aliases_to_add', input.aliases_to_add ?? [], term.term.source_language, aliasErrors)
        if (aliasErrors.length > 0) {
            return { ok: false, reason: aliasErrors.map(error => error.message).join('；') }
        }

        addAliasesToTerm(term.term, input.aliases_to_add ?? [])
    } else if (input.operation === 'change_term_type') {
        if (!input.term_type) {
            return { ok: false, reason: 'cached change_term_type missing term_type' }
        }
        term.term.term_type = input.term_type
        normalizeTermGenderPresentations(state, term.term)
    } else if (input.operation === 'deprecate_term') {
        term.term.status = 'deprecated'
        clearRejectedReason(term.term)
    } else if (input.operation === 'reject_term') {
        rejectTerm(term.term, input.rejected_reason ?? undefined)
    } else if (input.operation === 'merge_term') {
        if (!input.merged_into) {
            return { ok: false, reason: 'cached merge_term missing merged_into' }
        }
        const targetTerm = resolveCacheTargetTerm(state, input.merged_into)
        if (!targetTerm.ok) {
            return targetTerm
        }
        term.term.status = 'merged'
        term.term.merged_into = targetTerm.term.term_id
        clearRejectedReason(term.term)
    } else if (input.operation === 'append_gender_presentation') {
        if (!input.gender_presentation) {
            return { ok: false, reason: 'cached append_gender_presentation missing gender_presentation' }
        }
        const genderErrors = validateGenderPresentationBinding(state, term.term, input.gender_presentation, 'gender_presentation')
        if (genderErrors.length > 0) {
            return { ok: false, reason: `cached gender presentation validation failed: ${genderErrors.map(error => `${error.field}: ${error.message}`).join('; ')}` }
        }
        term.term.gender_presentations = [...(term.term.gender_presentations ?? []), input.gender_presentation]
    }

    term.term.revision += 1
    term.term.updated_at = now
    markTermUpdatedFromBatch(term.term, readCreatedBatchId(input.updated_from), readUpdatedMetadataFields(input))
    return { ok: true }
}

async function applyCachedUpdateTermEntries (
    root: string,
    manualTransFile: string,
    state: GlossaryState,
    input: UpdateTermEntriesInput,
): Promise<{ ok: true }|{ ok: false, reason: string }> {
    const term = resolveCacheTargetTerm(state, input.term_id)

    if (!term.ok) {
        return term
    }

    if (input.operation !== 'append_entry') {
        return { ok: false, reason: `unsupported cached entry operation: ${input.operation}` }
    }

    const requireEvidence = state.active_plan_id
        ? state.plans.find(plan => plan.plan_id === state.active_plan_id)?.term_extraction_policy.require_evidence ?? true
        : true

    if (requireEvidence && input.evidence.length === 0 && !allowsEvidenceOptionalTranslationRule(input.entry)) {
        return { ok: false, reason: 'cached append_entry missing required evidence' }
    }

    const genderErrors = validateNewGenderPresentationBinding(term.term, input)
    if (genderErrors.length > 0) {
        return { ok: false, reason: `cached gender presentation validation failed: ${genderErrors.map(error => `${error.field}: ${error.message}`).join('; ')}` }
    }

    const manualTrans = input.evidence.length > 0 ? await loadManualTransData(root, manualTransFile).catch(() => null) : null
    if (input.evidence.length > 0 && !manualTrans) {
        return { ok: false, reason: `cached append_entry could not load ${manualTransFile}` }
    }

    const normalizedEvidence = manualTrans
        ? await normalizeEvidencePayloads(root, input.evidence, manualTrans, undefined, createFullEvidenceRange(manualTrans))
        : { evidence: [] as NormalizedEvidencePayload[], errors: [] as ValidationError[] }
    if (normalizedEvidence.errors.length > 0) {
        return cacheValidationFailure(normalizedEvidence.errors)
    }

    appendEntryDirectly(state, term.term, input, normalizedEvidence.evidence)
    return { ok: true }
}

async function applyCachedMergeProposal (
    root: string,
    manualTransFile: string,
    state: GlossaryState,
    input: CreateTermMergeProposalInput,
): Promise<{ ok: true }|{ ok: false, reason: string }> {
    const sourceTerm = resolveCacheTargetTerm(state, input.source_term_id)
    const targetTerm = resolveCacheTargetTerm(state, input.target_term_id)

    if (!sourceTerm.ok) {
        return sourceTerm
    }
    if (!targetTerm.ok) {
        return targetTerm
    }

    const duplicateProposal = state.merge_proposals.find(proposal => (
        termPairKey(proposal.source_term_id, proposal.target_term_id) === termPairKey(sourceTerm.term.term_id, targetTerm.term.term_id)
        && ['candidate', 'approved', 'applied'].includes(proposal.status)
    ))

    if (duplicateProposal) {
        return { ok: true }
    }

    const manualTrans = input.evidence.length > 0 ? await loadManualTransData(root, manualTransFile).catch(() => null) : null
    if (input.evidence.length > 0 && !manualTrans) {
        return { ok: false, reason: `cached create_term_merge_proposal could not load ${manualTransFile}` }
    }

    const normalizedEvidence = manualTrans
        ? await normalizeEvidencePayloads(root, input.evidence, manualTrans, undefined, createFullEvidenceRange(manualTrans))
        : { evidence: [] as NormalizedEvidencePayload[], errors: [] as ValidationError[] }
    if (normalizedEvidence.errors.length > 0) {
        return cacheValidationFailure(normalizedEvidence.errors)
    }

    const now = new Date().toISOString()
    state.merge_proposals.push({
        proposal_id: createNextMergeProposalId(state),
        source_term_id: sourceTerm.term.term_id,
        target_term_id: targetTerm.term.term_id,
        status: 'candidate',
        reason: input.reason,
        evidence: normalizedEvidence.evidence,
        existing_evidence_ids: uniqueStrings(input.existing_evidence_ids),
        created_by: input.created_by,
        created_from: input.created_from,
        revision: 1,
        created_at: now,
        updated_at: now,
    })

    return { ok: true }
}

function appendEntryDirectly (state: GlossaryState, term: GlossaryTerm, input: UpdateTermEntriesInput, evidence: NormalizedEvidencePayload[]): void {
    const now = new Date().toISOString()
    const entryId = createNextEntryId(state, term.term_id)
    const evidenceIds = createNextEvidenceIds(state, term.term_id, input.evidence.length)
    const normalizedEntry = normalizeEntryForStorage(term, input.entry, 'entry')

    if (!normalizedEntry.ok) {
        throw new Error(`cached append entry source selector validation failed: ${JSON.stringify(normalizedEntry.errors)}`)
    }

    state.entries.push(buildGlossaryEntryFromPayload(normalizedEntry.entry, {
        entry_id: entryId,
        term_id: term.term_id,
        evidence_ids: evidenceIds,
        created_by: input.created_by,
        created_from: input.created_from,
        revision: 1,
        created_at: now,
        updated_at: now,
    }))
    state.evidence.push(...evidence.map((item, index): GlossaryEvidence => ({
        evidence_id: evidenceIds[index],
        term_id: term.term_id,
        entry_id: entryId,
        source_ref: {
            source_file_id: item.source_ref.source_file_id,
            key_index: item.source_ref.key_index,
            key_hash: item.source_ref.key_hash ? normalizeKeyHashForStorage(item.source_ref.key_hash) : undefined,
            span: item.source_ref.span,
        },
        quote: item.quote,
        context: item.context,
        reason: item.reason,
        created_by: input.created_by,
        created_at: now,
    })))
    term.entry_ids = uniqueStrings([...term.entry_ids, entryId])
    if (input.gender_presentation) {
        term.gender_presentations = [
            ...(term.gender_presentations ?? []),
            {
                ...input.gender_presentation,
                entry_id: entryId,
            },
        ]
    }
    term.revision += 1
    term.updated_at = now
    markTermUpdatedFromBatch(term, readCreatedBatchId(input.created_from), [
        'entry_ids',
        ...(input.gender_presentation ? ['gender_presentations'] : []),
    ])
}

function resolveCacheTargetTerm (state: GlossaryState, termId: string): { ok: true, term: GlossaryTerm }|{ ok: false, reason: string } {
    const term = state.terms.find(item => item.term_id === termId)

    if (!term) {
        return { ok: false, reason: `cached write references missing term ${termId}` }
    }

    if (term.status === 'merged' && term.merged_into) {
        const mergedInto = state.terms.find(item => item.term_id === term.merged_into)
        return mergedInto && mergedInto.status === 'active'
            ? { ok: true, term: mergedInto }
            : { ok: false, reason: `cached write references merged term ${termId} with invalid target ${term.merged_into}` }
    }

    if (term.status === 'rejected' || term.status === 'deprecated') {
        return { ok: false, reason: `cached write references ${term.status} term ${termId}` }
    }

    return { ok: true, term }
}

function cacheValidationFailure (errors: ValidationError[]): { ok: false, reason: string } {
    return {
        ok: false,
        reason: `cached write validation failed: ${errors.map(error => `${error.field}: ${error.message}`).join('; ')}`,
    }
}

function createFullEvidenceRange (manualTrans: ManualTransData): EvidenceRange[] {
    return [{
        source_file_id: manualTrans.relativePath,
        start_index: 0,
        end_index: Math.max(0, manualTrans.keys.length - 1),
    }]
}

async function createFilteredIndexByRawIndex (root: string, manualTrans: ManualTransData): Promise<Map<number, number>> {
    let keyItems: Awaited<ReturnType<typeof filterManualTransKeyItems>>
    try {
        keyItems = await filterManualTransKeyItems(root, manualTrans.keys, DEFAULT_SOURCE_LANGUAGE)
    } catch {
        keyItems = manualTrans.keys.map((key, index) => ({
            key,
            originalIndex: index,
            filteredIndex: index,
        }))
    }
    return new Map(keyItems.map(item => [item.originalIndex, item.filteredIndex]))
}

function validateNewGenderPresentationBinding (term: GlossaryTerm, input: UpdateTermEntriesInput): ValidationError[] {
    if (!input.gender_presentation) {
        return []
    }

    const errors: ValidationError[] = []

    if (term.term_type !== 'character') {
        errors.push({ field: 'gender_presentation', message: 'gender_presentation 只允许绑定到 character Term。' })
    }

    if (!isGenderPresentationEntry(input.entry)) {
        errors.push({ field: 'gender_presentation', message: 'gender_presentation 必须绑定 fact、style 或 continuity Entry。' })
    }

    return errors
}

function validateGenderPresentationBinding (
    state: GlossaryState,
    term: GlossaryTerm,
    presentation: GenderPresentation,
    field: string,
): ValidationError[] {
    return validateGenderPresentationsForTerm(state, term, presentation ? [presentation] : [], field, term.gender_presentations ?? [])
}

function validateGenderPresentationsForTerm (
    state: GlossaryState,
    term: GlossaryTerm,
    presentations: GenderPresentation[],
    field: string,
    existingPresentations: GenderPresentation[] = [],
): ValidationError[] {
    const errors: ValidationError[] = []

    if (presentations.length === 0) {
        return errors
    }

    if (term.term_type !== 'character') {
        errors.push({ field, message: 'gender_presentations 只适用于 character Term。' })
    }

    const entryIds = new Set(existingPresentations.map(item => item.entry_id))

    presentations.forEach((presentation, index) => {
        const presentationField = presentations.length === 1 ? field : `${field}[${index}]`
        const entry = state.entries.find(item => item.entry_id === presentation.entry_id)

        if (entryIds.has(presentation.entry_id)) {
            errors.push({ field: `${presentationField}.entry_id`, message: '同一个 Entry 最多只能绑定一个 gender_presentation。' })
            return
        }
        entryIds.add(presentation.entry_id)

        if (!entry) {
            errors.push({ field: `${presentationField}.entry_id`, message: 'gender_presentation.entry_id 指向的 Entry 不存在。' })
            return
        }

        if (entry.term_id !== term.term_id) {
            errors.push({ field: `${presentationField}.entry_id`, message: 'gender_presentation.entry_id 必须指向同一个 Term 下的 Entry。' })
        }

        if (!isGenderPresentationEntry(entry)) {
            errors.push({ field: `${presentationField}.entry_id`, message: 'gender_presentation 必须绑定 fact、style 或 continuity Entry。' })
        }
    })

    return errors
}

function isGenderPresentationEntry (entry: Pick<GlossaryEntry|EntryPayload, 'entry_type'>): boolean {
    return entry.entry_type === 'fact' || entry.entry_type === 'style' || entry.entry_type === 'continuity'
}

function normalizeTermGenderPresentations (state: GlossaryState, term: GlossaryTerm): void {
    if (!term.gender_presentations) {
        return
    }

    if (term.gender_presentations.length === 0) {
        delete term.gender_presentations
        return
    }

    if (term.term_type !== 'character') {
        delete term.gender_presentations
        return
    }

    const seenEntryIds = new Set<string>()
    term.gender_presentations = term.gender_presentations.filter(presentation => {
        if (seenEntryIds.has(presentation.entry_id)) {
            return false
        }

        const entry = state.entries.find(item => item.entry_id === presentation.entry_id)
        const valid = !!entry && entry.term_id === term.term_id && isGenderPresentationEntry(entry)

        if (valid) {
            seenEntryIds.add(presentation.entry_id)
        }

        return valid
    })

    if (term.gender_presentations.length === 0) {
        delete term.gender_presentations
    }
}

function removeGenderPresentationForEntry (state: GlossaryState, entryId: string): void {
    for (const term of state.terms) {
        if (!term.gender_presentations?.some(item => item.entry_id === entryId)) {
            continue
        }

        term.gender_presentations = term.gender_presentations.filter(item => item.entry_id !== entryId)
        if (term.gender_presentations.length === 0) {
            delete term.gender_presentations
        }
    }
}

function transferGenderPresentationBetweenEntries (
    state: GlossaryState,
    sourceEntry: GlossaryEntry,
    targetEntry: GlossaryEntry,
): void {
    const sourceTerm = state.terms.find(item => item.term_id === sourceEntry.term_id)
    const targetTerm = state.terms.find(item => item.term_id === targetEntry.term_id)
    const sourcePresentation = sourceTerm?.gender_presentations?.find(item => item.entry_id === sourceEntry.entry_id)

    removeGenderPresentationForEntry(state, sourceEntry.entry_id)

    if (!sourcePresentation || !targetTerm || targetTerm.term_type !== 'character' || !isGenderPresentationEntry(targetEntry)) {
        return
    }

    if (targetTerm.gender_presentations?.some(item => item.entry_id === targetEntry.entry_id)) {
        return
    }

    targetTerm.gender_presentations = [
        ...(targetTerm.gender_presentations ?? []),
        {
            ...sourcePresentation,
            entry_id: targetEntry.entry_id,
        },
    ]
    normalizeTermGenderPresentations(state, targetTerm)
}

function moveEntryToTerm (state: GlossaryState, entry: GlossaryEntry, targetTermId: string, remapSourceSelectors = true): void {
    const sourceTerm = state.terms.find(item => item.term_id === entry.term_id)
    const targetTerm = state.terms.find(item => item.term_id === targetTermId)

    if (!targetTerm) {
        addReviewConflict(state, 'review_entries_batch', `target Term ${targetTermId} 不存在。`, { entry_id: entry.entry_id, target_term_id: targetTermId })
        return
    }

    if (sourceTerm && remapSourceSelectors) {
        remapEntrySourceSelectorsForMove(entry, sourceTerm, targetTerm)
    }

    const movedGenderPresentations = sourceTerm?.gender_presentations?.filter(item => item.entry_id === entry.entry_id) ?? []
    if (sourceTerm && movedGenderPresentations.length > 0) {
        sourceTerm.gender_presentations = sourceTerm.gender_presentations?.filter(item => item.entry_id !== entry.entry_id)
        normalizeTermGenderPresentations(state, sourceTerm)
    }

    if (sourceTerm) {
        sourceTerm.entry_ids = sourceTerm.entry_ids.filter(entryId => entryId !== entry.entry_id)
    }

    targetTerm.entry_ids = uniqueStrings([...targetTerm.entry_ids, entry.entry_id])
    entry.term_id = targetTerm.term_id
    const now = new Date().toISOString()

    if (sourceTerm) {
        sourceTerm.revision += 1
        sourceTerm.updated_at = now
    }
    targetTerm.revision += 1
    targetTerm.updated_at = now

    for (const evidence of state.evidence.filter(item => item.entry_id === entry.entry_id)) {
        evidence.term_id = targetTerm.term_id
    }

    if (targetTerm.term_type === 'character' && isGenderPresentationEntry(entry)) {
        const existingEntryIds = new Set(targetTerm.gender_presentations?.map(item => item.entry_id) ?? [])
        targetTerm.gender_presentations = [
            ...(targetTerm.gender_presentations ?? []),
            ...movedGenderPresentations.filter(item => !existingEntryIds.has(item.entry_id)),
        ]
    }

    normalizeTermGenderPresentations(state, targetTerm)
}

function mergeTermIntoTarget (state: GlossaryState, sourceTerm: GlossaryTerm, targetTermId: string): void {
    const targetTerm = state.terms.find(item => item.term_id === targetTermId)

    if (!targetTerm || targetTerm.status !== 'active') {
        addReviewConflict(state, 'review_terms_batch', `merge target ${targetTermId} 不存在或不是 active。`, { term_id: sourceTerm.term_id, target_term_id: targetTermId })
        return
    }

    ensureAliasForText(targetTerm, sourceTerm.source_text)
    for (const aliasText of getAliasTexts(sourceTerm)) {
        ensureAliasForText(targetTerm, aliasText)
    }

    for (const entryId of [...sourceTerm.entry_ids]) {
        const entry = state.entries.find(item => item.entry_id === entryId)
        if (entry) {
            moveEntryToTerm(state, entry, targetTerm.term_id)
        }
    }

    sourceTerm.status = 'merged'
    sourceTerm.merged_into = targetTerm.term_id
    clearRejectedReason(sourceTerm)
    targetTerm.revision += 1
    targetTerm.updated_at = new Date().toISOString()

    for (const proposal of state.merge_proposals.filter(item => termPairKey(item.source_term_id, item.target_term_id) === termPairKey(sourceTerm.term_id, targetTerm.term_id))) {
        proposal.status = 'applied'
        proposal.revision += 1
        proposal.updated_at = new Date().toISOString()
    }
}

function remapEntrySourceSelectorsForMove (entry: GlossaryEntry, _sourceTerm: GlossaryTerm, targetTerm: GlossaryTerm): void {
    if (entry.entry_type !== 'translation_rule' || !entry.applicability.source_selectors) {
        return
    }

    entry.applicability.source_selectors = entry.applicability.source_selectors.map(selector => {
        if (selector.text === targetTerm.source_text) {
            return {
                variant_id: 'term',
                text: selector.text,
            }
        }

        return {
            variant_id: ensureAliasForText(targetTerm, selector.text),
            text: selector.text,
        }
    })
}

function findReferencedAliasIds (state: GlossaryState, term: GlossaryTerm, aliasTexts: string[]): string[] {
    const aliasIds = aliasTexts
        .map(aliasText => findAliasIdByText(term, aliasText))
        .filter((aliasId): aliasId is string => aliasId !== null)

    return aliasIds.filter(aliasId => state.entries.some(entry => entry.term_id === term.term_id && entryReferencesAliasId(entry, aliasId)))
}

function addReviewConflict (
    state: GlossaryState,
    toolName: string,
    reason: string,
    input?: Record<string, unknown>,
    cacheId?: string,
): void {
    state.review.conflicts.push({
        conflict_id: createNextReviewConflictId(state),
        cache_id: cacheId,
        tool_name: toolName,
        reason,
        input,
        created_at: new Date().toISOString(),
    })
}

function recordSnapshot (session: GlossaryToolSession): void {
    if (session.snapshots.length === 0) {
        session.snapshots.push({ dataRevision: 'current' })
    }
}

function recordQueriedTerm (session: GlossaryToolSession, term: GlossaryTerm): void {
    session.queriedTermIds.add(term.term_id)
    session.queriedTermRevisions.set(term.term_id, term.revision)
}

function refreshSessionSnapshots (session: GlossaryToolSession, _state: GlossaryState): void {
    session.snapshots = session.snapshots.map(() => ({ dataRevision: 'current' }))
}

function recordQueriedSourceKey (session: GlossaryToolSession, text: string, sourceLanguage: string, matchModes: MatchMode[]): void {
    const key = sourceKey(text, sourceLanguage)
    const modes = session.queriedSourceKeys.get(key) ?? new Set<MatchMode>()

    for (const mode of matchModes) {
        modes.add(mode)
    }

    session.queriedSourceKeys.set(key, modes)
}

function hasSourceQueryForCreate (session: GlossaryToolSession, sourceText: string, sourceLanguage: string): boolean {
    const modes = session.queriedSourceKeys.get(sourceKey(sourceText, sourceLanguage))
    return !!modes && ['exact', 'alias', 'compound'].some(mode => modes.has(mode as MatchMode))
}

function createTermRetryQuery (term: GlossaryTerm|undefined): unknown {
    if (!term) {
        return {}
    }

    return createSourceTextRetryQuery(term.source_text, term.source_language)
}

function createSourceTextRetryQuery (sourceText: string, sourceLanguage: string): unknown {
    return {
        queries: [
            {
                text: sourceText,
                match_modes: ['exact', 'alias', 'compound'],
            },
        ],
        source_language: sourceLanguage,
        include_entries: true,
        include_evidence: true,
        term_statuses: ['active', 'merged', 'deprecated', 'rejected'],
        limit: 20,
    }
}

function mergeProposalMatchesFilter (
    proposal: GlossaryMergeProposal,
    input: QueryTermMergeProposalsInput,
    statuses: Set<string>|null,
): boolean {
    if (statuses && !statuses.has(proposal.status)) {
        return false
    }

    if (input.source_term_id && input.target_term_id) {
        return termPairKey(proposal.source_term_id, proposal.target_term_id) === termPairKey(input.source_term_id, input.target_term_id)
    }

    if (input.source_term_id) {
        return proposal.source_term_id === input.source_term_id || proposal.target_term_id === input.source_term_id
    }

    if (input.target_term_id) {
        return proposal.source_term_id === input.target_term_id || proposal.target_term_id === input.target_term_id
    }

    return true
}

function findExactDuplicateTerm (state: GlossaryState, input: CreateOrGetTermInput): GlossaryTerm|null {
    const candidateKeys = new Set([input.source_text, ...input.aliases].map(normalizeSourceText))

    return state.terms.find(term => {
        if (term.source_language !== input.source_language || term.status !== 'active' && term.status !== 'merged') {
            return false
        }

        return [term.source_text, ...getAliasTexts(term)].some(value => candidateKeys.has(normalizeSourceText(value)))
    }) ?? null
}

function findPossibleDuplicateTerms (state: GlossaryState, input: CreateOrGetTermInput): GlossaryTerm[] {
    const candidateKeys = [input.source_text, ...input.aliases].map(normalizeSourceText)
    const duplicates: GlossaryTerm[] = []

    for (const term of state.terms) {
        if (term.source_language !== input.source_language || term.status !== 'active' && term.status !== 'merged') {
            continue
        }

        const termKeys = [term.source_text, ...getAliasTexts(term)].map(normalizeSourceText)
        const hasPossibleMatch = candidateKeys.some(candidateKey => termKeys.some(termKey => (
            candidateKey.includes(termKey)
            || termKey.includes(candidateKey)
            || similarity(candidateKey, termKey) >= 0.72
        )))

        if (hasPossibleMatch) {
            duplicates.push(term)
        }
    }

    return duplicates.slice(0, 10)
}

function createPossibleDuplicateRetryQuery (input: CreateOrGetTermInput, terms: GlossaryTerm[]): unknown {
    return {
        queries: [
            {
                text: input.source_text,
                match_modes: ['exact', 'alias', 'compound', 'fuzzy'],
            },
            ...terms.map(term => ({
                text: term.source_text,
                match_modes: ['exact', 'alias', 'compound', 'fuzzy'],
            })),
        ],
        source_language: input.source_language,
        include_entries: true,
        include_evidence: true,
        term_statuses: ['active', 'merged', 'deprecated', 'rejected'],
        limit: 20,
    }
}

function validateMergeProposalTerms (sourceTerm: GlossaryTerm|undefined, targetTerm: GlossaryTerm|undefined): ValidationError[] {
    const errors: ValidationError[] = []

    if (!sourceTerm) {
        errors.push({ field: 'source_term_id', message: 'source_term_id 指向的 Term 不存在。' })
    } else if (sourceTerm.status !== 'active') {
        errors.push({ field: 'source_term_id', message: 'source_term_id 必须指向 active Term。' })
    }

    if (!targetTerm) {
        errors.push({ field: 'target_term_id', message: 'target_term_id 指向的 Term 不存在。' })
    } else if (targetTerm.status !== 'active') {
        errors.push({ field: 'target_term_id', message: 'target_term_id 必须指向 active Term。' })
    }

    return errors
}

function validateExistingEvidenceIdsForProposal (
    evidenceIds: string[],
    state: GlossaryState,
    session: GlossaryToolSession,
): ValidationError[] {
    const errors: ValidationError[] = []

    evidenceIds.forEach((evidenceId, index) => {
        if (!session.returnedEvidenceIds.has(evidenceId)) {
            errors.push({
                field: `existing_evidence_ids[${index}]`,
                message: 'existing_evidence_ids 只能引用当前 agent 已通过 query/search 返回的 Evidence。',
            })
            return
        }

        if (!state.evidence.some(evidence => evidence.evidence_id === evidenceId)) {
            errors.push({
                field: `existing_evidence_ids[${index}]`,
                message: 'existing_evidence_ids 包含不存在的 Evidence。',
            })
        }
    })

    return errors
}

function termPairKey (leftTermId: string, rightTermId: string): string {
    return [leftTermId, rightTermId].sort().join('\0')
}

// Same ManualTrans object contract as translationStore.loadManualTransData.
async function loadManualTransData (root: string, manualTransFile: string): Promise<ManualTransData> {
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

function createContextKeyItemsByRawIndexes (
    manualTrans: ManualTransData,
    rawIndexes: number[],
    maxCharsPerKey: number,
    includeValues: boolean,
    filteredIndexByRawIndex: Map<number, number>,
): unknown[] {
    return rawIndexes
        .map(index => {
            const filteredKeyIndex = filteredIndexByRawIndex.get(index)
            if (filteredKeyIndex === undefined) {
                return null
            }

            const text = manualTrans.keys[index]
            const truncatedText = truncate(text, maxCharsPerKey)
            const value = manualTrans.values[index]
            const item: Record<string, unknown> = {
                filtered_key_index: filteredKeyIndex,
                text: truncatedText,
                truncated: truncatedText.length < text.length,
            }

            if (includeValues) {
                const valueText = typeof value === 'string' ? value : JSON.stringify(value)
                item.value = truncate(valueText, maxCharsPerKey)
            }

            return item
        })
        .filter((item): item is Record<string, unknown> => item !== null)
}

function createNextTermId (state: GlossaryState): string {
    const nextNumber = state.terms.reduce((max, term) => Math.max(max, readTrailingNumber(term.term_id, /^term_(\d+)$/)), 0) + 1
    return `term_${nextNumber.toString().padStart(6, '0')}`
}

function createNextMergeProposalId (state: GlossaryState): string {
    const nextNumber = state.merge_proposals.reduce((max, proposal) => Math.max(max, readTrailingNumber(proposal.proposal_id, /^merge_proposal_(\d+)$/)), 0) + 1
    return `merge_proposal_${nextNumber.toString().padStart(6, '0')}`
}

function createNextReviewWindowId (state: GlossaryState): string {
    const nextNumber = state.review.windows.reduce((max, window) => Math.max(max, readTrailingNumber(window.review_window_id, /^review_window_(\d+)$/)), 0) + 1
    return `review_window_${nextNumber.toString().padStart(6, '0')}`
}

function createNextCacheId (state: GlossaryState): string {
    const nextNumber = state.review.cached_worker_writes.reduce((max, write) => Math.max(max, readTrailingNumber(write.cache_id, /^cache_write_(\d+)$/)), 0) + 1
    return `cache_write_${nextNumber.toString().padStart(6, '0')}`
}

function createNextReviewConflictId (state: GlossaryState): string {
    const nextNumber = state.review.conflicts.reduce((max, conflict) => Math.max(max, readTrailingNumber(conflict.conflict_id, /^review_conflict_(\d+)$/)), 0) + 1
    return `review_conflict_${nextNumber.toString().padStart(6, '0')}`
}

function createNextReviewActionId (nextNumber: number): string {
    return `review_action_${nextNumber.toString().padStart(4, '0')}`
}

function createNextEntryId (state: GlossaryState, termId: string): string {
    const termNumber = readTrailingNumber(termId, /^term_(\d+)$/)
    const prefix = `entry_${termNumber.toString().padStart(6, '0')}_`
    const nextNumber = state.entries.reduce((max, entry) => {
        if (!entry.entry_id.startsWith(prefix)) {
            return max
        }

        return Math.max(max, readTrailingNumber(entry.entry_id, new RegExp(`^${prefix}(\\d+)$`)))
    }, 0) + 1

    return `${prefix}${nextNumber.toString().padStart(2, '0')}`
}

function createNextEvidenceIds (state: GlossaryState, termId: string, count: number): string[] {
    const termNumber = readTrailingNumber(termId, /^term_(\d+)$/)
    const prefix = `evi_${termNumber.toString().padStart(6, '0')}_`
    const firstNumber = state.evidence.reduce((max, evidence) => {
        if (!evidence.evidence_id.startsWith(prefix)) {
            return max
        }

        return Math.max(max, readTrailingNumber(evidence.evidence_id, new RegExp(`^${prefix}(\\d+)$`)))
    }, 0) + 1

    return Array.from({ length: count }, (_, index) => `${prefix}${(firstNumber + index).toString().padStart(3, '0')}`)
}

function readTrailingNumber (value: string, regex: RegExp): number {
    const match = regex.exec(value)
    return match ? Number(match[1]) : 0
}

function valuesForEntryField (entry: GlossaryEntry, field: EntryTextField): string[] {
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

    return entry.applicability.does_not_apply_when ?? []
}

function normalizeEntryFilter (filter: SearchGlossaryEntriesInput['entry_filter']): EntryFilter|null {
    if (!filter) {
        return null
    }

    return {
        entry_types: filter.entry_types ?? null,
        entry_statuses: filter.entry_statuses ?? null,
        applies_to: filter.applies_to ?? null,
        target_language: filter.target_language ?? null,
        policy_strength: filter.policy_strength ?? null,
        domain: filter.domain ?? null,
        text: filter.text ?? null,
        text_fields: filter.text_fields ?? null,
        is_regex: filter.is_regex,
        case_sensitive: filter.case_sensitive,
    }
}

function formatBatchId (batchNumber: number): string {
    return `batch_${batchNumber.toString().padStart(4, '0')}`
}

function readCreatedBatchId (createdFrom: Record<string, unknown>|undefined): string {
    return typeof createdFrom?.batch_id === 'string' ? createdFrom.batch_id : ''
}

function hasTermUpdatedFromAnyBatch (term: GlossaryTerm, batchIds: Set<string>): boolean {
    const directBatchId = readCreatedBatchId(term.updated_from)

    if (directBatchId && batchIds.has(directBatchId)) {
        return true
    }

    const fieldBatchIds = term.updated_from?.field_batch_ids
    if (!isRecord(fieldBatchIds)) {
        return false
    }

    return Object.values(fieldBatchIds).some(batchId => typeof batchId === 'string' && batchIds.has(batchId))
}

function markTermUpdatedFromBatch (term: GlossaryTerm, batchId: string|undefined, fields: string[] = []): void {
    if (!batchId) {
        return
    }

    const existingFieldBatchIds = isRecord(term.updated_from?.field_batch_ids)
        ? term.updated_from.field_batch_ids
        : {}

    term.updated_from = {
        ...(term.updated_from ?? {}),
        batch_id: batchId,
        field_batch_ids: {
            ...existingFieldBatchIds,
            ...Object.fromEntries(fields.map(field => [field, batchId])),
        },
    }
}

function readUpdatedMetadataFields (input: UpdateTermMetadataInput): string[] {
    if (input.operation === 'add_aliases') {
        return ['aliases', 'alias_order', 'next_alias_seq']
    }

    if (input.operation === 'change_term_type') {
        return ['term_type', 'gender_presentations']
    }

    if (input.operation === 'deprecate_term') {
        return ['status', 'rejected_reason']
    }

    if (input.operation === 'reject_term') {
        return ['status', 'rejected_reason']
    }

    if (input.operation === 'merge_term') {
        return ['status', 'merged_into', 'rejected_reason']
    }

    if (input.operation === 'append_gender_presentation') {
        return ['gender_presentations']
    }

    return []
}

function readCacheTermId (input: Record<string, unknown>): string|null {
    if (typeof input.term_id === 'string') {
        return input.term_id
    }

    if (Array.isArray(input.items)) {
        const firstItem = input.items.find(isRecord)
        return typeof firstItem?.term_id === 'string' ? firstItem.term_id : null
    }

    return null
}

function normalizeTermFilter (filter: SearchGlossaryEntriesInput['term_filter']): TermFilter|null {
    if (!filter) {
        return null
    }

    return {
        source_language: filter.source_language ?? null,
        term_types: filter.term_types ?? null,
        term_statuses: filter.term_statuses ?? null,
    }
}

function normalizeUpdateTermMetadataInput (input: UpdateTermMetadataInput): UpdateTermMetadataInput {
    return {
        ...input,
        aliases_to_add: input.aliases_to_add ?? null,
        term_type: input.term_type ?? null,
        merged_into: input.merged_into ?? null,
        gender_presentation: input.gender_presentation ?? null,
        rejected_reason: input.rejected_reason ?? null,
    }
}

function compactUpdateTermMetadataInputForCache (input: UpdateTermMetadataInput): Record<string, unknown> {
    const compact: Record<string, unknown> = { ...input }

    for (const field of ['aliases_to_add', 'term_type', 'merged_into', 'gender_presentation', 'rejected_reason']) {
        if (compact[field] === null || compact[field] === undefined) {
            delete compact[field]
        }
    }

    return compact
}

function normalizeUpdateTermEntriesInput (input: UpdateTermEntriesInput): UpdateTermEntriesInput {
    return {
        ...input,
        entry_id: input.entry_id ?? null,
        expected_entry_revision: input.expected_entry_revision ?? null,
        gender_presentation: input.gender_presentation ?? null,
        evidence: input.evidence ?? [],
    }
}

// Differs from translation text search: non-regex matching only folds case.
function createTextMatcher (text: string, isRegex: boolean, caseSensitive: boolean): (value: string) => boolean {
    if (isRegex) {
        const regex = new RegExp(text, caseSensitive ? '' : 'i')
        return value => {
            regex.lastIndex = 0
            return regex.test(value)
        }
    }

    const needle = caseSensitive ? text : text.toLowerCase()
    return value => (caseSensitive ? value : value.toLowerCase()).includes(needle)
}

function validateRegexQueries (queries: QueryGlossaryTermsInput['queries']): ValidationError[] {
    const errors: ValidationError[] = []

    queries.forEach((query, index) => {
        if (query.match_modes.includes('regex')) {
            errors.push(...validateRegexText(`queries[${index}].text`, query.text))
        }
    })

    return errors
}

// Same regex compile check as translation tools; returns localized validation errors.
function validateRegexText (field: string, text: string): ValidationError[] {
    try {
        new RegExp(text)
        return []
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        return [{ field, message: `无效正则表达式: ${message}` }]
    }
}

// Same text normalization contract as translation matching, named for source-term keys.
function normalizeSourceText (value: string): string {
    return normalizeWhitespace(value).toLocaleLowerCase()
}

function normalizeWhitespace (value: string): string {
    return value.normalize('NFKC').trim().replace(/\s+/g, ' ')
}

function sourceKey (sourceText: string, sourceLanguage: string): string {
    return `${sourceLanguage}\0${normalizeSourceText(sourceText)}`
}

function normalizeKeyHash (value: string): string {
    return value.startsWith('sha256:') ? value.slice('sha256:'.length) : value
}

function normalizeKeyHashForStorage (value: string): string {
    return value.startsWith('sha256:') ? value : `sha256:${value}`
}

// Same raw SHA-256 helper as translationTools.sha256Text.
function sha256Text (value: string): string {
    return createHash('sha256').update(value).digest('hex')
}

// Same stored source-hash contract as translationStore: sha256: plus hex.
function sha256SourceText (value: string): string {
    return `sha256:${sha256Text(value)}`
}

// Same similarity helper as translationTools.similarity.
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

function hasEffectiveEntryFilter (filter: EntryFilter|null): filter is EntryFilter {
    return !!filter && (
        !!filter.entry_types
        || !!filter.entry_statuses
        || !!filter.applies_to
        || !!filter.target_language
        || !!filter.policy_strength
        || !!filter.domain
        || !!filter.text
    )
}

function hasEffectiveTermFilter (filter: TermFilter|null): filter is TermFilter {
    return !!filter && (!!filter.source_language || !!filter.term_types || !!filter.term_statuses)
}

function isEvidenceKeyIndexAllowed (
    sourceFileId: string,
    keyIndex: number,
    filteredKeyIndex: number,
    evidenceScope: EvidenceScope|undefined,
    allowedEvidenceRanges: EvidenceRange[],
): boolean {
    const normalizedSourceFileId = normalizeProjectFileId(sourceFileId)

    if (
        evidenceScope
        && normalizeProjectFileId(evidenceScope.source_file_id) === normalizedSourceFileId
        && evidenceScope.allowed_key_indices
        && evidenceScope.allowed_key_indices.includes(keyIndex)
    ) {
        return true
    }

    if (
        evidenceScope
        && normalizeProjectFileId(evidenceScope.source_file_id) === normalizedSourceFileId
        && filteredKeyIndex >= Math.max(0, evidenceScope.batch_start_index - evidenceScope.context_window)
        && filteredKeyIndex <= evidenceScope.batch_end_index + evidenceScope.context_window
    ) {
        return true
    }

    return allowedEvidenceRanges.some(range => {
        if (normalizeProjectFileId(range.source_file_id) !== normalizedSourceFileId) {
            return false
        }

        if (range.allowed_key_indices) {
            return range.allowed_key_indices.includes(keyIndex)
        }

        return keyIndex >= range.start_index && keyIndex <= range.end_index
    })
}

function getEvidenceSourceFileId (evidence: GlossaryEvidence): string|null {
    return evidence.source_ref?.source_file_id ?? evidence.source_ref?.file_id ?? null
}

// Dead code: submit_glossary_plan no longer rejects Term/Entry/Evidence-shaped content heuristically.
function rejectGlossaryRecords (value: unknown, field: string, errors: ValidationError[]): void {
    if (Array.isArray(value)) {
        value.forEach((item, index) => rejectGlossaryRecords(item, `${field}[${index}]`, errors))
        return
    }

    if (!isRecord(value)) {
        return
    }

    if (looksLikeTermRecord(value)) {
        errors.push({
            field,
            message: 'submit_glossary_plan 不能包含 Term 记录。',
        })
    }

    if (looksLikeEntryRecord(value)) {
        errors.push({
            field,
            message: 'submit_glossary_plan 不能包含 Entry 记录。',
        })
    }

    if (looksLikeEvidenceRecord(value)) {
        errors.push({
            field,
            message: 'submit_glossary_plan 不能包含 Evidence 记录。',
        })
    }

    for (const [key, childValue] of Object.entries(value)) {
        rejectGlossaryRecords(childValue, `${field}.${key}`, errors)
    }
}

// Dead code: submit_glossary_plan no longer rejects concrete-translation-looking object keys heuristically.
function rejectConcreteTranslationHints (value: unknown, field: string, errors: ValidationError[]): void {
    if (Array.isArray(value)) {
        value.forEach((item, index) => rejectConcreteTranslationHints(item, `${field}[${index}]`, errors))
        return
    }

    if (!isRecord(value)) {
        return
    }

    const forbiddenKeys = [
        'preferred_translation',
        'alternative_translations',
        'forbidden_translations',
        'target',
        'evidence',
    ]

    for (const key of forbiddenKeys) {
        if (Object.prototype.hasOwnProperty.call(value, key)) {
            errors.push({
                field: `${field}.${key}`,
                message: 'submit_glossary_plan 不允许包含具体译名、禁用译名或 Evidence/target 结构。',
            })
        }
    }

    for (const [key, childValue] of Object.entries(value)) {
        rejectConcreteTranslationHints(childValue, `${field}.${key}`, errors)
    }
}

// Dead code: submit_glossary_plan no longer scans plan strings for forbidden literal text.
function rejectForbiddenPlanText (value: unknown, field: string, errors: ValidationError[]): void {
    if (Array.isArray(value)) {
        value.forEach((item, index) => rejectForbiddenPlanText(item, `${field}[${index}]`, errors))
        return
    }

    if (typeof value === 'string') {
        if (containsSchedulingInstruction(value)) {
            errors.push({
                field,
                message: 'submit_glossary_plan 不允许输出 batching、batch_size 或 parallel_agents 等调度策略。',
            })
        }

        if (containsConcreteTranslationHint(value)) {
            errors.push({
                field,
                message: 'submit_glossary_plan 不允许包含具体译名、禁用译名或正式译名建议。',
            })
        }

        return
    }

    if (!isRecord(value)) {
        return
    }

    for (const [key, childValue] of Object.entries(value)) {
        rejectForbiddenPlanText(childValue, `${field}.${key}`, errors)
    }
}

// Dead code: retained only as the old submit_glossary_plan literal-string screen.
function containsSchedulingInstruction (value: string): boolean {
    return /\b(batching|batch_size|batch size|parallel_agents|parallel agents)\b/iu.test(value)
}

// Dead code: retained only as the old submit_glossary_plan literal-string screen.
function containsConcreteTranslationHint (value: string): boolean {
    return /(禁用译名|禁译|译为|翻译为|preferred_translation|alternative_translations|forbidden_translations)/iu.test(value)
}

function looksLikeTermRecord (value: Record<string, unknown>): boolean {
    return typeof value.term_id === 'string'
        && typeof value.source_text === 'string'
        && typeof value.term_type === 'string'
}

function looksLikeEntryRecord (value: Record<string, unknown>): boolean {
    return typeof value.entry_id === 'string'
        && typeof value.term_id === 'string'
        && typeof value.entry_type === 'string'
}

function looksLikeEvidenceRecord (value: Record<string, unknown>): boolean {
    return typeof value.evidence_id === 'string'
        && typeof value.term_id === 'string'
        && typeof value.entry_id === 'string'
}

function validateExactSet<T extends readonly string[]> (
    field: string,
    values: string[]|null,
    requiredValues: T,
    errors: ValidationError[],
): void {
    if (!values) {
        errors.push({
            field,
            message: `${field} 必须为 ${requiredValues.join(' 和 ')}。`,
        })
        return
    }

    const requiredSet = new Set<string>(requiredValues)
    const valueSet = new Set<string>(values)
    const missingValues = requiredValues.filter(value => !valueSet.has(value))
    const invalidValues = values.filter(value => !requiredSet.has(value))

    if (values.length !== requiredValues.length || missingValues.length > 0 || invalidValues.length > 0) {
        errors.push({
            field,
            message: `${field} 必须且只能包含 ${requiredValues.join(', ')}。`,
        })
    }
}

// Dead code: submit_glossary_plan no longer rejects generic/template-looking list text.
function validateUsefulList (field: string, values: string[]|null, errors: ValidationError[]): void {
    if (!values || values.length === 0) {
        errors.push({
            field,
            message: `${field} 至少需要 1 条，并且应基于 inspection tools 的结果填写。`,
        })
        return
    }

    if (values.every(value => isGenericTemplateText(value))) {
        errors.push({
            field,
            message: `${field} 内容过于通用，应体现 inspection tools 的统计、样本或命中结果。`,
        })
    }
}

// Dead code: retained only as the old submit_glossary_plan generic/template text screen.
function isGenericTemplateText (value: string): boolean {
    const normalizedValue = value.trim().toLowerCase()

    return normalizedValue.length < 4
        || normalizedValue === 'todo'
        || normalizedValue === 'n/a'
        || normalizedValue === 'none'
        || normalizedValue.includes('待填写')
        || normalizedValue.includes('模板')
        || normalizedValue.includes('通用')
}

function validateAliases (field: string, aliases: string[], sourceLanguage: string, errors: ValidationError[]): void {
    if (sourceLanguage === DEFAULT_SOURCE_LANGUAGE) {
        return
    }

    const invalidAliases = aliases.filter(alias => looksLikeTargetLanguage(alias))

    if (invalidAliases.length > 0) {
        errors.push({
            field,
            message: `${field} 不允许包含明显目标语言译名: ${invalidAliases.join(', ')}。`,
            code: 'invalid_alias',
        })
    }
}

function looksLikeTargetLanguage (value: string): boolean {
    return /[\u4e00-\u9fff]/u.test(value) && !/[\u3040-\u30ff]/u.test(value)
}

// Same tool payload string handling as translation tools: trim, drop empty values, keep first-seen order.
function uniqueStrings (values: string[]): string[] {
    return Array.from(new Set(values.map(value => value.trim()).filter(Boolean)))
}

function hasOwn (value: Record<string, unknown>, key: string): boolean {
    return Object.prototype.hasOwnProperty.call(value, key)
}

// Same plain-object guard as agent/taskStore/glossaryStore/translationStore isRecord.
function isRecord (value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
}
