import { existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'

const nodeRequire = createRequire(import.meta.url)

const PACKAGE_RESOURCES_DIRECTORY_NAME = 'resources'
const DEFAULT_LANGUAGE_MODEL_RELATIVE_PATH = path.join('data', 'models', 'mediapipe', 'language_detector.tflite')

export function resolveKuromojiDictionaryPath (): string {
    return resolveExistingPath([
        () => path.join(resolvePackageResourcesRoot(), 'kuromoji', 'dict'),
        () => path.join(path.dirname(nodeRequire.resolve('kuromoji/package.json')), 'dict'),
    ], 'Unable to locate kuromoji dictionary directory.')
}

export function resolveMediaPipeWasmPath (): string {
    return resolveExistingPath([
        () => path.join(resolvePackageResourcesRoot(), 'mediapipe', 'wasm'),
        () => path.join(path.dirname(nodeRequire.resolve('@mediapipe/tasks-text')), 'wasm'),
    ], 'Unable to locate MediaPipe text wasm directory.')
}

export function resolveLanguageModelPath (root: string): string {
    return resolveExistingPath([
        () => process.env.TRANS20_LANGUAGE_DETECTOR_MODEL,
        () => path.join(resolvePackageResourcesRoot(), 'mediapipe', 'language_detector.tflite'),
        () => path.join(root, DEFAULT_LANGUAGE_MODEL_RELATIVE_PATH),
        () => path.resolve(root, '..', DEFAULT_LANGUAGE_MODEL_RELATIVE_PATH),
        () => path.resolve(root, '..', '..', DEFAULT_LANGUAGE_MODEL_RELATIVE_PATH),
    ], [
        'MediaPipe language detector model not found.',
        'Set TRANS20_LANGUAGE_DETECTOR_MODEL,',
        `place language_detector.tflite under ${DEFAULT_LANGUAGE_MODEL_RELATIVE_PATH},`,
        'or include resources/mediapipe/language_detector.tflite next to the packaged executable.',
    ].join(' '))
}

function resolvePackageResourcesRoot (): string {
    return path.join(path.dirname(process.execPath), PACKAGE_RESOURCES_DIRECTORY_NAME)
}

function resolveExistingPath (candidates: Array<() => string|undefined>, message: string): string {
    const checkedPaths: string[] = []

    for (const candidate of candidates) {
        const candidatePath = safelyResolve(candidate)

        if (!candidatePath) {
            continue
        }

        checkedPaths.push(candidatePath)

        if (existsSync(candidatePath)) {
            return candidatePath
        }
    }

    throw new Error(`${message} Checked: ${checkedPaths.join(', ')}`)
}

function safelyResolve (candidate: () => string|undefined): string|undefined {
    try {
        return candidate()
    } catch {
        return undefined
    }
}
