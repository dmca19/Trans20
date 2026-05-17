import path from 'node:path'

import { z } from 'zod'

import { readOptionalJsonFile, writeJsonFile } from './fileUtils.js'
import { DEFAULT_TUI_LANGUAGE, TUI_LANGUAGE_VALUES, type TuiLanguage } from './tuiLanguage.js'

export const preferencesSchema = z.object({
    tuiLanguage: z.enum(TUI_LANGUAGE_VALUES).optional(),
    setupWizardCompleted: z.boolean().optional(),
})

export type Preferences = {
    tuiLanguage: TuiLanguage
    setupWizardCompleted: boolean
}

export type RawPreferences = z.infer<typeof preferencesSchema>

export async function loadPreferences (): Promise<Preferences> {
    return mergePreferences(await readPreferencesFile())
}

export async function loadRawPreferences (): Promise<RawPreferences> {
    return readPreferencesFile()
}

export async function savePreferences (preferences: Preferences): Promise<void> {
    const preferencesPath = getPreferencesPath()
    const validatedPreferences = preferencesSchema.required().parse(preferences)
    await writeJsonFile(preferencesPath, validatedPreferences)
}

export async function saveTuiLanguagePreference (tuiLanguage: TuiLanguage): Promise<void> {
    const rawPreferences = await readPreferencesFile()
    await savePreferences(mergePreferences({
        ...rawPreferences,
        tuiLanguage,
    }))
}

export function mergePreferences (rawPreferences: RawPreferences): Preferences {
    return {
        tuiLanguage: rawPreferences.tuiLanguage ?? DEFAULT_TUI_LANGUAGE,
        setupWizardCompleted: rawPreferences.setupWizardCompleted ?? false,
    }
}

function getPreferencesPath (): string {
    return path.join(process.cwd(), 'preferences.json')
}

async function readPreferencesFile (): Promise<RawPreferences> {
    const preferencesPath = getPreferencesPath()
    return readOptionalJsonFile(preferencesPath, preferencesSchema, {})
}
