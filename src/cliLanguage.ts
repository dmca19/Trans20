import { getSupportedTuiLanguages, isTuiLanguage, type TuiLanguage } from './tuiLanguage.js'

export type TuiLanguageArgumentResult =
    | { ok: true, language: TuiLanguage|null }
    | { ok: false, error: string }

export function parseTuiLanguageArgument (args: string[]): TuiLanguageArgumentResult {
    for (let index = 0; index < args.length; index += 1) {
        const arg = args[index]

        if (!arg) {
            continue
        }

        if (arg === '--language') {
            const value = args[index + 1]

            return value
                ? parseTuiLanguageValue(value)
                : invalidTuiLanguageArgument(arg)
        }

        if (arg.startsWith('--language=') || arg === '--tui-language' || arg.startsWith('--tui-language=') || arg === '-l') {
            return invalidTuiLanguageFormat(arg)
        }
    }

    return { ok: true, language: null }
}

function parseTuiLanguageValue (value: string): TuiLanguageArgumentResult {
    return isTuiLanguage(value)
        ? { ok: true, language: value }
        : invalidTuiLanguageArgument(value)
}

function invalidTuiLanguageArgument (value: string): TuiLanguageArgumentResult {
    return {
        ok: false,
        error: `Invalid TUI language "${value}". Supported values: ${getSupportedTuiLanguages().join(', ')}. / 无效的 TUI 语言 "${value}"。支持的值：${getSupportedTuiLanguages().join(', ')}。`,
    }
}

function invalidTuiLanguageFormat (value: string): TuiLanguageArgumentResult {
    return {
        ok: false,
        error: `Invalid TUI language argument "${value}". Use --language <value>. Supported values: ${getSupportedTuiLanguages().join(', ')}. / 无效的 TUI 语言参数 "${value}"。请使用 --language <值>。支持的值：${getSupportedTuiLanguages().join(', ')}。`,
    }
}
