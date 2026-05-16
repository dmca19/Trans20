import type { EntryStatus, GlossaryEntry, GlossaryTerm } from './glossaryStore.js'

export type CharacterContextWarning = {
    code: 'high_value_character_without_gender_presentation'
    severity: 'warning'
    priority: 'light'
    field: 'gender_presentations'
    term_id: string
    source_text: string
    entry_count: number
    alias_count: number
    message: string
}

const HIGH_VALUE_CHARACTER_ENTRY_COUNT = 3
const HIGH_VALUE_CHARACTER_ALIAS_COUNT = 3
const DEFAULT_CHARACTER_CONTEXT_WARNING_LIMIT = 10

export function collectCharacterContextWarnings (input: {
    terms: GlossaryTerm[]
    entries: GlossaryEntry[]
    termIds?: Set<string>|null
    approvedOnly?: boolean
    limit?: number
}): CharacterContextWarning[] {
    const allowedStatuses: EntryStatus[] = input.approvedOnly ? ['approved'] : ['approved', 'candidate']
    const allowedStatusSet = new Set<EntryStatus>(allowedStatuses)
    const warnings = input.terms
        .filter(term => !input.termIds || input.termIds.has(term.term_id))
        .filter(term => term.status === 'active' && term.term_type === 'character')
        .map(term => {
            const entries = input.entries.filter(entry => entry.term_id === term.term_id)
            const entryCount = entries.filter(entry => allowedStatusSet.has(entry.status)).length
            const aliasCount = countAliases(term)

            return {
                term,
                entries,
                entryCount,
                aliasCount,
            }
        })
        .filter(item => item.entryCount >= HIGH_VALUE_CHARACTER_ENTRY_COUNT || item.aliasCount >= HIGH_VALUE_CHARACTER_ALIAS_COUNT)
        .filter(item => !hasValidGenderPresentation(item.term, item.entries, allowedStatusSet))
        .sort((left, right) => (right.entryCount + right.aliasCount) - (left.entryCount + left.aliasCount) || left.term.term_id.localeCompare(right.term.term_id))
        .slice(0, input.limit ?? DEFAULT_CHARACTER_CONTEXT_WARNING_LIMIT)
        .map(item => formatCharacterContextWarning(item.term, item.entryCount, item.aliasCount, input.approvedOnly === true))

    return warnings
}

export function hasValidGenderPresentation (
    term: GlossaryTerm,
    entries: GlossaryEntry[],
    allowedStatusSet = new Set<EntryStatus>(['approved', 'candidate']),
): boolean {
    return (term.gender_presentations ?? []).some(presentation => {
        const entry = entries.find(item => item.entry_id === presentation.entry_id)

        return !!entry
            && entry.term_id === term.term_id
            && allowedStatusSet.has(entry.status)
            && isGenderPresentationEntry(entry)
    })
}

function formatCharacterContextWarning (
    term: GlossaryTerm,
    entryCount: number,
    aliasCount: number,
    approvedOnly: boolean,
): CharacterContextWarning {
    const coverageText = approvedOnly ? 'approved gender_presentations' : 'candidate or approved gender_presentations'

    return {
        code: 'high_value_character_without_gender_presentation',
        severity: 'warning',
        priority: 'light',
        field: 'gender_presentations',
        term_id: term.term_id,
        source_text: term.source_text,
        entry_count: entryCount,
        alias_count: aliasCount,
        message: `Light warning: high-value character ${term.term_id} ${term.source_text} has no ${coverageText}. If text evidence affects gender/address translation, add a dedicated fact/style/continuity Entry and bind gender_presentation; low confidence is acceptable for directional but uncertain evidence. Do not write unknown/low just to clear this warning; defer with reason when evidence is insufficient.`,
    }
}

function isGenderPresentationEntry (entry: Pick<GlossaryEntry, 'entry_type'>): boolean {
    return entry.entry_type === 'fact' || entry.entry_type === 'style' || entry.entry_type === 'continuity'
}

function countAliases (term: GlossaryTerm): number {
    return new Set(Object.values(term.aliases).map(alias => alias.trim()).filter(Boolean)).size
}
