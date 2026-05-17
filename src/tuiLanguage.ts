export const TUI_LANGUAGE_VALUES = ['en', 'zh-CN'] as const

export type TuiLanguage = typeof TUI_LANGUAGE_VALUES[number]

export const DEFAULT_TUI_LANGUAGE: TuiLanguage = 'en'

export const TUI_LANGUAGE_PROMPT_MESSAGE = 'Select TUI language / 选择界面语言'

const TUI_LANGUAGE_CHOICE_LABELS = {
    en: {
        name: 'English',
        short: 'English',
    },
    'zh-CN': {
        name: '中文',
        short: '中文',
    },
} satisfies Record<TuiLanguage, {
    name: string
    short: string
}>

export const TUI_LANGUAGE_CHOICES: ReadonlyArray<{
    value: TuiLanguage
    name: string
    short: string
}> = TUI_LANGUAGE_VALUES.map(value => ({
    value,
    ...TUI_LANGUAGE_CHOICE_LABELS[value],
}))

export function getDefaultTuiLanguage (): TuiLanguage {
    return DEFAULT_TUI_LANGUAGE
}

export function getTuiLanguageChoices (): typeof TUI_LANGUAGE_CHOICES {
    return TUI_LANGUAGE_CHOICES
}

export function isTuiLanguage (value: string): value is TuiLanguage {
    return TUI_LANGUAGE_CHOICES.some(choice => choice.value === value)
}

export function getSupportedTuiLanguages (): TuiLanguage[] {
    return [...TUI_LANGUAGE_VALUES]
}

export function isTuiLanguagePreferenceMissing (rawPreferences: { tuiLanguage?: TuiLanguage }): boolean {
    return rawPreferences.tuiLanguage === undefined
}
