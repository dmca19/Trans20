import { findAliasIdByText, getSourceVariants } from './glossaryAliasUtils.js'

import type {
    GlossaryEntry,
    GlossaryReviewWindow,
    GlossaryState,
    GlossaryTerm,
    SourceSelector,
    TermStatus,
    TermType,
} from './glossaryStore.js'
import type { CheckTranslationRuleCoverageInput } from './glossaryToolTypes.js'

export type GlossaryCoverageIssue = {
    code:
        'character_name_without_approved_translation_rule'
        |'character_alias_without_translation_rule'
        |'sole_name_rule_rejected'
        |'item_without_translation_rule'
        |'repeated_phrase_without_translation_rule'
        |'system_term_without_translation_rule'
        |'fact_only_translatable_term'
    severity: 'warning'|'blocking'
    term_id: string
    source_text: string
    variant_id?: string
    variant_index?: number
    variant_text?: string
    covered_by_entry_ids?: string[]
    candidate_entry_ids?: string[]
    rejected_entry_ids?: string[]
    message: string
}

export type CheckTranslationRuleCoverageOutput = {
    ok: true
    issues: GlossaryCoverageIssue[]
    summary: {
        blocking_count: number
        warning_count: number
    }
}

const PROPER_NAME_COVERAGE_TERM_TYPE_VALUES: Array<'character'|'faction'|'place'|'title'> = [
    'character',
    'faction',
    'place',
    'title',
]

const DEFAULT_COVERAGE_TERM_TYPES: TermType[] = [
    ...PROPER_NAME_COVERAGE_TERM_TYPE_VALUES,
    'item',
    'repeated_phrase',
    'system_term',
]

const PROPER_NAME_COVERAGE_TERM_TYPES = new Set<TermType>(PROPER_NAME_COVERAGE_TERM_TYPE_VALUES)

export function collectTranslationRuleCoverageIssues (
    state: GlossaryState,
    input: CheckTranslationRuleCoverageInput,
): CheckTranslationRuleCoverageOutput {
    const window = findReviewWindow(state, input.review_window_id)
    const termTypes = new Set<TermType>(input.term_types ?? DEFAULT_COVERAGE_TERM_TYPES)
    const explicitTermIds = input.term_ids ? new Set(input.term_ids) : null
    const reviewTermIds = explicitTermIds ?? collectReviewWindowTermIds(state, window)
    const issues: GlossaryCoverageIssue[] = []

    for (const term of state.terms) {
        const effectiveTerm = applyPendingTermActions(term, window)

        if (!reviewTermIds.has(term.term_id) || effectiveTerm.status !== 'active' || !termTypes.has(effectiveTerm.term_type)) {
            continue
        }

        const allEntries = collectEffectiveEntriesForTerm(state, window, term)

        if (!hasEffectiveEntry(allEntries)) {
            continue
        }

        const variants = getSourceVariants(term)
        const entries = allEntries.filter(entry => entry.entry_type === 'translation_rule')
        const approvedEntries = entries.filter(entry => entry.status === 'approved')
        const candidateEntries = entries.filter(entry => entry.status === 'candidate')
        const rejectedEntries = entries.filter(entry => entry.status === 'rejected')
        const approvedNonRuleEntries = allEntries.filter(entry => (
            entry.entry_type !== 'translation_rule'
            && entry.status === 'approved'
        ))
        const approvedVariantIds = collectCoveredVariantIds(term, approvedEntries)
        const effectiveRuleVariantIds = collectCoveredVariantIds(term, [...approvedEntries, ...candidateEntries])
        const approvedEntryIds = approvedEntries.map(entry => entry.entry_id)
        const rejectedEntryIds = rejectedEntries.map(entry => entry.entry_id)

        if (PROPER_NAME_COVERAGE_TERM_TYPES.has(effectiveTerm.term_type)) {
            collectProperNameCoverageIssues({
                issues,
                input,
                term,
                variants,
                approvedVariantIds,
                approvedEntryIds,
                candidateEntries,
                rejectedEntries,
                rejectedEntryIds,
                approvedEntries,
            })
            continue
        }

        if ((effectiveTerm.term_type === 'item' || effectiveTerm.term_type === 'repeated_phrase') && approvedNonRuleEntries.length > 0 && approvedEntries.length === 0) {
            issues.push({
                code: 'fact_only_translatable_term',
                severity: 'warning',
                term_id: term.term_id,
                source_text: term.source_text,
                variant_id: 'term',
                variant_index: 0,
                variant_text: term.source_text,
                covered_by_entry_ids: approvedNonRuleEntries.map(entry => entry.entry_id),
                message: `${term.term_id} ${term.source_text} has approved fact/style/continuity Entries but no approved translation_rule for source_variant_indexes[0].`,
            })
            continue
        }

        if (effectiveRuleVariantIds.has('term')) {
            continue
        }

        const code = nonProperNameMissingRuleCode(effectiveTerm.term_type)
        if (code) {
            issues.push({
                code,
                severity: 'warning',
                term_id: term.term_id,
                source_text: term.source_text,
                variant_id: 'term',
                variant_index: 0,
                variant_text: term.source_text,
                ...(input.include_rejected_candidates ? nonEmptyEntryIds('rejected_entry_ids', collectEntryIdsCoveringVariant('term', rejectedEntries)) : {}),
                message: `${term.term_id} ${term.source_text} has no candidate or approved translation_rule for source_variant_indexes[0].`,
            })
        }
    }

    return {
        ok: true,
        issues,
        summary: {
            blocking_count: issues.filter(issue => issue.severity === 'blocking').length,
            warning_count: issues.filter(issue => issue.severity === 'warning').length,
        },
    }
}

function collectProperNameCoverageIssues (context: {
    issues: GlossaryCoverageIssue[]
    input: CheckTranslationRuleCoverageInput
    term: GlossaryTerm
    variants: ReturnType<typeof getSourceVariants>
    approvedVariantIds: Set<string>
    approvedEntryIds: string[]
    candidateEntries: GlossaryEntry[]
    rejectedEntries: GlossaryEntry[]
    rejectedEntryIds: string[]
    approvedEntries: GlossaryEntry[]
}): void {
    const {
        issues,
        input,
        term,
        variants,
        approvedVariantIds,
        approvedEntryIds,
        candidateEntries,
        rejectedEntries,
        rejectedEntryIds,
        approvedEntries,
    } = context

    if (!approvedVariantIds.has('term')) {
        const candidateEntryIds = collectEntryIdsCoveringVariant('term', candidateEntries)
        const rejectedTermEntryIds = collectEntryIdsCoveringVariant('term', rejectedEntries)
        issues.push({
            code: 'character_name_without_approved_translation_rule',
            severity: 'blocking',
            term_id: term.term_id,
            source_text: term.source_text,
            variant_id: 'term',
            variant_index: 0,
            variant_text: term.source_text,
            ...(approvedEntryIds.length > 0 ? { covered_by_entry_ids: approvedEntryIds } : {}),
            ...(candidateEntryIds.length > 0 ? { candidate_entry_ids: candidateEntryIds } : {}),
            ...(input.include_rejected_candidates && rejectedTermEntryIds.length > 0 ? { rejected_entry_ids: rejectedTermEntryIds } : {}),
            message: `${term.term_id} ${term.source_text} has no approved translation_rule for source_variant_indexes[0].`,
        })
    }

    for (const variant of variants.filter(item => item.index > 0)) {
        if (approvedVariantIds.has(variant.variant_id)) {
            continue
        }

        issues.push({
            code: 'character_alias_without_translation_rule',
            severity: 'warning',
            term_id: term.term_id,
            source_text: term.source_text,
            variant_id: variant.variant_id,
            variant_index: variant.index,
            variant_text: variant.text,
            ...(approvedEntryIds.length > 0 ? { covered_by_entry_ids: approvedEntryIds } : {}),
            ...nonEmptyEntryIds('candidate_entry_ids', collectEntryIdsCoveringVariant(variant.variant_id, candidateEntries)),
            ...(input.include_rejected_candidates ? nonEmptyEntryIds('rejected_entry_ids', collectEntryIdsCoveringVariant(variant.variant_id, rejectedEntries)) : {}),
            message: `${term.term_id} alias ${variant.text} is not covered by any approved translation_rule.`,
        })
    }

    if (input.include_rejected_candidates && rejectedEntryIds.length > 0 && approvedEntries.length === 0) {
        issues.push({
            code: 'sole_name_rule_rejected',
            severity: 'blocking',
            term_id: term.term_id,
            source_text: term.source_text,
            rejected_entry_ids: rejectedEntryIds,
            message: `${term.term_id} has rejected translation_rule candidates but no approved replacement.`,
        })
    }
}

function findReviewWindow (state: GlossaryState, reviewWindowId: string): GlossaryReviewWindow|null {
    if (state.review.active_window?.review_window_id === reviewWindowId) {
        return state.review.active_window
    }

    return state.review.windows.find(window => window.review_window_id === reviewWindowId) ?? null
}

function hasEffectiveEntry (entries: GlossaryEntry[]): boolean {
    return entries.some(entry => entry.status === 'approved' || entry.status === 'candidate')
}

function nonProperNameMissingRuleCode (termType: TermType): GlossaryCoverageIssue['code']|null {
    if (termType === 'item') {
        return 'item_without_translation_rule'
    }
    if (termType === 'repeated_phrase') {
        return 'repeated_phrase_without_translation_rule'
    }
    if (termType === 'system_term') {
        return 'system_term_without_translation_rule'
    }

    return null
}

function collectReviewWindowTermIds (state: GlossaryState, window: GlossaryReviewWindow|null): Set<string> {
    if (!window) {
        return new Set()
    }

    const batchIds = new Set(window.completed_batch_numbers.map(formatBatchId))
    const termIds = new Set<string>()

    for (const term of state.terms) {
        if (batchIds.has(readCreatedBatchId(term.created_from)) || batchIds.has(readCreatedBatchId(term.updated_from))) {
            termIds.add(term.term_id)
        }
    }

    for (const entry of state.entries) {
        if (batchIds.has(readCreatedBatchId(entry.created_from))) {
            termIds.add(entry.term_id)
        }
    }

    for (const action of window.pending_entry_actions) {
        const entry = state.entries.find(item => item.entry_id === action.entry_id)
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

function applyPendingTermActions (term: GlossaryTerm, window: GlossaryReviewWindow|null): Pick<GlossaryTerm, 'term_type'|'status'> {
    let termType: TermType = term.term_type
    let status: TermStatus = term.status

    for (const action of window?.pending_term_actions ?? []) {
        if (action.term_id !== term.term_id) {
            continue
        }

        if (action.operation === 'reject') {
            status = 'rejected'
        } else if (action.operation === 'deprecate') {
            status = 'deprecated'
        } else if (action.operation === 'merge_term') {
            status = 'merged'
        } else if (action.operation === 'change_term_type' && action.term_type) {
            termType = action.term_type
        } else if (action.operation === 'keep_active') {
            status = 'active'
        }
    }

    return { term_type: termType, status }
}

function collectEffectiveEntriesForTerm (state: GlossaryState, window: GlossaryReviewWindow|null, term: GlossaryTerm): GlossaryEntry[] {
    return [
        ...state.entries.map(entry => {
            const originalTerm = state.terms.find(item => item.term_id === entry.term_id) ?? term
            return applyPendingEntryActions(entry, window, originalTerm, state.terms)
        }),
        ...collectPendingAppendEntries(term, window),
    ].filter(entry => entry.term_id === term.term_id)
}

function applyPendingEntryActions (
    entry: GlossaryEntry,
    window: GlossaryReviewWindow|null,
    term: GlossaryTerm,
    terms: GlossaryTerm[] = [term],
): GlossaryEntry {
    let effective = cloneGlossaryEntry(entry)

    for (const action of window?.pending_entry_actions ?? []) {
        if (action.entry_id !== entry.entry_id) {
            continue
        }

        if (action.operation === 'approve') {
            effective.status = 'approved'
        } else if (action.operation === 'reject' || action.operation === 'merge_into') {
            effective.status = 'rejected'
        } else if ((action.operation === 'move_to_term' || action.operation === 'move_to_term_and_approve') && action.target_term_id) {
            const targetTerm = terms.find(item => item.term_id === action.target_term_id) ?? term
            const hasRevisedSourceVariantIndexes = Array.isArray(action.revised_entry?.applicability?.source_variant_indexes)
            effective = cloneGlossaryEntry({
                ...effective,
                term_id: action.target_term_id,
                ...(action.operation === 'move_to_term_and_approve' ? { status: 'approved' as const } : {}),
            })
            if (!hasRevisedSourceVariantIndexes) {
                effective = remapPendingEntrySourceSelectorsForMove(effective, targetTerm)
            }

            if (action.revised_entry) {
                effective = applyPendingRevisedEntry(effective, action.revised_entry, targetTerm, effective.status)
            }
        } else if (action.operation === 'revise') {
            effective = applyPendingRevisedEntry(effective, action.revised_entry, term, 'approved')
        }
    }

    return effective
}

function cloneGlossaryEntry (entry: GlossaryEntry): GlossaryEntry {
    if (entry.entry_type === 'translation_rule') {
        return {
            ...entry,
            content: { ...entry.content },
            target: { ...entry.target },
            applicability: {
                ...entry.applicability,
                source_selectors: entry.applicability.source_selectors ? [...entry.applicability.source_selectors] : undefined,
            },
            policy: {
                ...entry.policy,
                notes: entry.policy.notes ? [...entry.policy.notes] : undefined,
            },
        }
    }

    return {
        ...entry,
        content: { ...entry.content },
        applicability: {
            ...entry.applicability,
            source_selectors: entry.applicability.source_selectors ? [...entry.applicability.source_selectors] : undefined,
        },
        policy: {
            ...entry.policy,
            notes: entry.policy.notes ? [...entry.policy.notes] : undefined,
        },
    }
}

function remapPendingEntrySourceSelectorsForMove (entry: GlossaryEntry, targetTerm: GlossaryTerm): GlossaryEntry {
    if (entry.entry_type !== 'translation_rule' || !entry.applicability.source_selectors) {
        return entry
    }

    return {
        ...entry,
        applicability: {
            ...entry.applicability,
            source_selectors: entry.applicability.source_selectors.map(selector => ({
                variant_id: selector.text === targetTerm.source_text
                    ? 'term'
                    : findAliasIdByText(targetTerm, selector.text) ?? `pending_alias:${selector.text}`,
                text: selector.text,
            })),
        },
    }
}

function applyPendingRevisedEntry (
    entry: GlossaryEntry,
    revisedEntry: GlossaryReviewWindow['pending_entry_actions'][number]['revised_entry'],
    term: GlossaryTerm,
    status: GlossaryEntry['status'],
): GlossaryEntry {
    if (!revisedEntry) {
        return {
            ...entry,
            status,
        }
    }

    return buildEffectiveEntry({
        ...entry,
        ...revisedEntry,
        content: revisedEntry.content ? { ...revisedEntry.content } : entry.content,
        target: revisedEntry.target ? { ...revisedEntry.target } : entry.target,
        applicability: revisedEntry.applicability
            ? normalizePendingApplicability(term, revisedEntry.applicability, entry.applicability)
            : entry.applicability,
        policy: revisedEntry.policy ? { ...revisedEntry.policy } : entry.policy,
        status,
    })
}

function collectPendingAppendEntries (term: GlossaryTerm, window: GlossaryReviewWindow|null): GlossaryEntry[] {
    return (window?.pending_entry_actions ?? [])
        .filter(action => action.operation === 'append' && action.term_id === term.term_id && action.entry)
        .map((action, index): GlossaryEntry => buildEffectiveEntry({
            ...action.entry!,
            entry_id: `pending:${action.action_id || index}`,
            term_id: term.term_id,
            status: 'approved',
            evidence_ids: action.evidence_ids ?? [],
            created_by: 'review-agent',
            created_from: { review_window_id: window?.review_window_id },
            revision: 1,
            created_at: '',
            updated_at: '',
        }))
}

function buildEffectiveEntry (entry: GlossaryEntry|NonNullable<GlossaryReviewWindow['pending_entry_actions'][number]['entry']> & {
    entry_id: string
    term_id: string
    status: GlossaryEntry['status']
    evidence_ids: string[]
    created_by?: string
    created_from?: Record<string, unknown>
    revision: number
    created_at: string
    updated_at: string
}): GlossaryEntry {
    const base = {
        entry_id: entry.entry_id,
        term_id: entry.term_id,
        content: entry.content,
        applicability: entry.applicability,
        policy: entry.policy,
        status: entry.status,
        evidence_ids: entry.evidence_ids,
        created_by: entry.created_by,
        created_from: entry.created_from,
        revision: entry.revision,
        created_at: entry.created_at,
        updated_at: entry.updated_at,
    }

    if (entry.entry_type === 'translation_rule') {
        if (!entry.basis || !entry.target) {
            throw new Error('effective translation_rule missing basis or target')
        }

        return {
            ...base,
            entry_type: 'translation_rule',
            basis: entry.basis,
            target: entry.target,
        }
    }

    return {
        ...base,
        entry_type: entry.entry_type,
    }
}

function normalizePendingApplicability (
    term: GlossaryTerm,
    applicability: GlossaryEntry['applicability'] & { source_variant_indexes?: number[] },
    fallback: GlossaryEntry['applicability'],
): GlossaryEntry['applicability'] {
    const {
        source_variant_indexes: _sourceVariantIndexes,
        ...cleanApplicability
    } = applicability

    const sourceSelectors = Array.isArray(applicability.source_variant_indexes)
        ? getSourceVariants(term)
            .filter(variant => applicability.source_variant_indexes?.includes(variant.index))
            .map(variant => ({
                variant_id: variant.variant_id,
                text: variant.text,
            }))
        : applicability.source_selectors ? [...applicability.source_selectors] : fallback.source_selectors

    return {
        ...fallback,
        ...cleanApplicability,
        source_selectors: sourceSelectors,
    }
}

function collectCoveredVariantIds (term: GlossaryTerm, entries: GlossaryEntry[]): Set<string> {
    const variants = getSourceVariants(term)
    const covered = new Set<string>()

    for (const entry of entries) {
        for (const selector of readEntrySourceSelectors(entry)) {
            const variant = variants.find(item => item.variant_id === selector.variant_id)
            if (variant && variant.text === selector.text) {
                covered.add(selector.variant_id)
            }
        }
    }

    return covered
}

function readEntrySourceSelectors (entry: GlossaryEntry): SourceSelector[] {
    return entry.applicability.source_selectors ?? []
}

function collectEntryIdsCoveringVariant (variantId: string, entries: GlossaryEntry[]): string[] {
    return entries
        .filter(entry => readEntrySourceSelectors(entry).some(selector => selector.variant_id === variantId))
        .map(entry => entry.entry_id)
}

function nonEmptyEntryIds (field: 'candidate_entry_ids'|'rejected_entry_ids', entryIds: string[]): Partial<Pick<GlossaryCoverageIssue, 'candidate_entry_ids'|'rejected_entry_ids'>> {
    return entryIds.length > 0 ? { [field]: entryIds } : {}
}

function formatBatchId (batchNumber: number): string {
    return `batch_${batchNumber.toString().padStart(4, '0')}`
}

function readCreatedBatchId (createdFrom: Record<string, unknown>|undefined): string {
    return typeof createdFrom?.batch_id === 'string' ? createdFrom.batch_id : ''
}
