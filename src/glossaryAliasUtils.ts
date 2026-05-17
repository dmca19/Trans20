import type { GlossaryEntry, GlossaryTerm, SourceSelector } from './glossaryStore.js'

export type AgentAliasView = {
    index: number
    text: string
}

export type SourceVariant = {
    index: number
    variant_id: 'term'|string
    text: string
}

export type AgentTermSourceView = {
    source_text_index: 0
    aliases: AgentAliasView[]
}

export function createAliasState (termId: string, aliases: string[]): Pick<GlossaryTerm, 'aliases'|'alias_order'|'next_alias_seq'> {
    const aliasMap: Record<string, string> = {}
    const aliasOrder: string[] = []
    let nextAliasSeq = 1

    for (const alias of uniqueTrimmed(aliases)) {
        const aliasId = createAliasId(termId, nextAliasSeq)
        aliasMap[aliasId] = alias
        aliasOrder.push(aliasId)
        nextAliasSeq += 1
    }

    return {
        aliases: aliasMap,
        alias_order: aliasOrder,
        next_alias_seq: nextAliasSeq,
    }
}

export function addAliasesToTerm (term: GlossaryTerm, aliases: string[]): void {
    const existingTexts = new Set(getAliasTexts(term))

    for (const alias of uniqueTrimmed(aliases)) {
        if (existingTexts.has(alias)) {
            continue
        }

        const aliasId = createAliasId(term.term_id, term.next_alias_seq)
        term.aliases[aliasId] = alias
        term.alias_order.push(aliasId)
        term.next_alias_seq += 1
        existingTexts.add(alias)
    }
}

export function removeAliasesFromTermByText (term: GlossaryTerm, aliasesToRemove: string[]): string[] {
    const removeTexts = new Set(aliasesToRemove.map(alias => alias.trim()).filter(Boolean))
    const removedAliasIds: string[] = []

    for (const aliasId of [...term.alias_order]) {
        const aliasText = term.aliases[aliasId]

        if (aliasText && removeTexts.has(aliasText)) {
            delete term.aliases[aliasId]
            removedAliasIds.push(aliasId)
        }
    }

    term.alias_order = term.alias_order.filter(aliasId => !!term.aliases[aliasId])
    return removedAliasIds
}

export function getAliasTexts (term: GlossaryTerm): string[] {
    return term.alias_order
        .map(aliasId => term.aliases[aliasId])
        .filter((alias): alias is string => typeof alias === 'string' && alias.trim().length > 0)
}

export function getSourceVariants (term: GlossaryTerm): SourceVariant[] {
    return [
        {
            index: 0,
            variant_id: 'term',
            text: term.source_text,
        },
        ...term.alias_order
            .map((aliasId, orderIndex): SourceVariant|null => {
                const text = term.aliases[aliasId]

                if (!text) {
                    return null
                }

                return {
                    index: orderIndex + 1,
                    variant_id: aliasId,
                    text,
                }
            })
            .filter((item): item is SourceVariant => item !== null),
    ]
}

export function formatTermSourceView (term: GlossaryTerm): AgentTermSourceView {
    return {
        source_text_index: 0,
        aliases: getSourceVariants(term)
            .filter(variant => variant.index > 0)
            .map(variant => ({
                index: variant.index,
                text: variant.text,
            })),
    }
}

export function formatTermForAgent (term: GlossaryTerm): Record<string, unknown> & { term_id: string } {
    return {
        term_id: term.term_id,
        source_text: term.source_text,
        ...formatTermSourceView(term),
        source_language: term.source_language,
        term_type: term.term_type,
        status: term.status,
        ...(term.rejected_reason ? { rejected_reason: term.rejected_reason } : {}),
        merged_into: term.merged_into,
        entry_ids: term.entry_ids,
        ...(term.gender_presentations ? { gender_presentations: term.gender_presentations } : {}),
        ...(term.created_by ? { created_by: term.created_by } : {}),
        ...(term.created_from ? { created_from: term.created_from } : {}),
        revision: term.revision,
        created_at: term.created_at,
        updated_at: term.updated_at,
    }
}

export function resolveSourceVariantIndexes (
    term: GlossaryTerm,
    indexes: number[]|undefined,
    field = 'applicability.source_variant_indexes',
): { ok: true, selectors: SourceSelector[] }|{ ok: false, errors: { field: string, message: string }[] } {
    const errors: { field: string, message: string }[] = []

    if (!indexes || indexes.length === 0) {
        return {
            ok: false,
            errors: [{ field, message: `${field} 必须为非空数组。` }],
        }
    }

    const variants = getSourceVariants(term)
    const variantsByIndex = new Map(variants.map(variant => [variant.index, variant]))
    const selectors: SourceSelector[] = []

    for (const [position, index] of indexes.entries()) {
        if (!Number.isInteger(index) || index < 0) {
            errors.push({ field: `${field}[${position}]`, message: 'source variant index 必须是非负整数。' })
            continue
        }

        const variant = variantsByIndex.get(index)

        if (!variant) {
            errors.push({ field: `${field}[${position}]`, message: `source variant index ${index} 不存在。` })
            continue
        }

        if (!selectors.some(selector => selector.variant_id === variant.variant_id)) {
            selectors.push({
                variant_id: variant.variant_id,
                text: variant.text,
            })
        }
    }

    return errors.length > 0 ? { ok: false, errors } : { ok: true, selectors }
}

export function selectorsToAgentApplicabilityFields (
    term: GlossaryTerm,
    selectors: SourceSelector[]|undefined,
    field = 'applicability.source_selectors',
): {
    source_variant_indexes?: number[]
    source_variant_texts?: string[]
    warnings: { code: 'stale_source_selector', field: string, message: string }[]
} {
    if (!selectors || selectors.length === 0) {
        return { warnings: [] }
    }

    const variants = getSourceVariants(term)
    const indexes: number[] = []
    const texts: string[] = []
    const warnings: { code: 'stale_source_selector', field: string, message: string }[] = []

    for (const [position, selector] of selectors.entries()) {
        const variant = variants.find(item => item.variant_id === selector.variant_id)

        if (!variant) {
            warnings.push({ code: 'stale_source_selector', field: `${field}[${position}]`, message: `source selector ${selector.variant_id} no longer exists.` })
            continue
        }

        if (variant.text !== selector.text) {
            warnings.push({ code: 'stale_source_selector', field: `${field}[${position}]`, message: `source selector ${selector.variant_id} text snapshot no longer matches.` })
            continue
        }

        indexes.push(variant.index)
        texts.push(variant.text)
    }

    return {
        source_variant_indexes: indexes,
        source_variant_texts: texts,
        warnings,
    }
}

export function entryReferencesAliasId (entry: GlossaryEntry, aliasId: string): boolean {
    return (entry.applicability.source_selectors ?? []).some(selector => selector.variant_id === aliasId)
}

export function findAliasIdByText (term: GlossaryTerm, text: string): string|null {
    const trimmedText = text.trim()

    return term.alias_order.find(aliasId => term.aliases[aliasId] === trimmedText) ?? null
}

export function ensureAliasForText (term: GlossaryTerm, text: string): string {
    const existingAliasId = findAliasIdByText(term, text)

    if (existingAliasId) {
        return existingAliasId
    }

    const aliasId = createAliasId(term.term_id, term.next_alias_seq)
    term.aliases[aliasId] = text
    term.alias_order.push(aliasId)
    term.next_alias_seq += 1
    return aliasId
}

export function formatEntryForAgent (entry: GlossaryEntry, term: GlossaryTerm): Record<string, unknown> {
    const selectorFields = selectorsToAgentApplicabilityFields(term, entry.applicability.source_selectors)
    const {
        source_selectors: _sourceSelectors,
        ...applicability
    } = entry.applicability

    return {
        entry_id: entry.entry_id,
        term_id: entry.term_id,
        entry_type: entry.entry_type,
        ...(entry.basis ? { basis: entry.basis } : {}),
        content: entry.content,
        ...(entry.target ? { target: entry.target } : {}),
        applicability: {
            ...applicability,
            ...(selectorFields.source_variant_indexes ? { source_variant_indexes: selectorFields.source_variant_indexes } : {}),
            ...(selectorFields.source_variant_texts ? { source_variant_texts: selectorFields.source_variant_texts } : {}),
        },
        policy: entry.policy,
        status: entry.status,
        evidence_ids: entry.evidence_ids,
        ...(entry.created_by ? { created_by: entry.created_by } : {}),
        ...(entry.created_from ? { created_from: entry.created_from } : {}),
        revision: entry.revision,
        created_at: entry.created_at,
        updated_at: entry.updated_at,
        ...(selectorFields.warnings.length > 0 ? {
            source_selector_warnings: selectorFields.warnings.map(warning => ({
                ...warning,
                entry_id: entry.entry_id,
            })),
        } : {}),
    }
}

function createAliasId (termId: string, seq: number): string {
    const normalizedTermId = termId.startsWith('term_') ? termId.slice('term_'.length) : termId
    return `alias_${normalizedTermId}_${seq.toString().padStart(2, '0')}`
}

function uniqueTrimmed (values: string[]): string[] {
    return Array.from(new Set(values.map(value => value.trim()).filter(Boolean)))
}
