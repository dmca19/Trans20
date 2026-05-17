import type { ComponentType } from 'react'
import type { AppProps } from './App.js'

if (!process.env.NODE_ENV) {
    process.env.NODE_ENV = 'production'
}

const [
    { select },
    { getDefaultTuiLanguage, getTuiLanguageChoices, isTuiLanguagePreferenceMissing, TUI_LANGUAGE_PROMPT_MESSAGE },
    { loadRawPreferences, mergePreferences, saveTuiLanguagePreference },
    { parseTuiLanguageArgument },
] = await Promise.all([
    import('@inquirer/prompts'),
    import('./tuiLanguage.js'),
    import('./preferences.js'),
    import('./cliLanguage.js'),
])

const rawPreferences = await loadRawPreferences()
const languageArgument = parseTuiLanguageArgument(process.argv.slice(2))
let language = mergePreferences(rawPreferences).tuiLanguage

if (!languageArgument.ok) {
    console.error(languageArgument.error)
    process.exit(1)
}

if (languageArgument.language) {
    language = languageArgument.language
} else if (isTuiLanguagePreferenceMissing(rawPreferences) && process.stdin.isTTY && process.stdout.isTTY) {
    language = await select({
        message: TUI_LANGUAGE_PROMPT_MESSAGE,
        choices: getTuiLanguageChoices(),
        default: getDefaultTuiLanguage(),
        loop: false,
    })
    await saveTuiLanguagePreference(language)
}

const [{ default: React }, { render }, { App }] = await Promise.all([
    import('react'),
    import('ink'),
    import('./App.js'),
])

const appElement = React.createElement(App as ComponentType<AppProps>, { language })

render(appElement, { alternateScreen: true, exitOnCtrlC: false })
