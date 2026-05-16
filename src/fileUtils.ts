import { mkdir, readFile } from 'node:fs/promises'
import path from 'node:path'

import writeFileAtomic from 'write-file-atomic'
import { z } from 'zod'

const DEFAULT_ATOMIC_WRITE_ATTEMPTS = 10
const RETRYABLE_ATOMIC_WRITE_CODES = new Set(['EPERM', 'EACCES', 'EBUSY'])

export type AtomicWriteFileOptions = {
    encoding?: BufferEncoding
    writeAttempts?: number
    fsync?: boolean
}

export async function atomicWriteFile (
    filePath: string,
    contents: string|NodeJS.ArrayBufferView,
    options: AtomicWriteFileOptions = {},
): Promise<void> {
    await mkdir(path.dirname(filePath), { recursive: true })

    const maxAttempts = options.writeAttempts ?? DEFAULT_ATOMIC_WRITE_ATTEMPTS

    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
        try {
            await writeFileAtomic(filePath, toWriteFileAtomicData(contents), {
                encoding: options.encoding ?? 'utf8',
                fsync: options.fsync ?? false,
            })
            return
        } catch (error) {
            if (!isRetryableAtomicWriteError(error) || attempt === maxAttempts) {
                throw error
            }

            await delay(Math.min(100 * attempt, 1_000))
        }
    }
}

export async function readOptionalJsonFile<T> (
    filePath: string,
    schema: z.ZodType<T>,
    missingValue: T,
): Promise<T> {
    try {
        const parsedValue = await readJsonFile(filePath)
        return schema.parse(parsedValue)
    } catch (error) {
        if (isNodeError(error) && error.code === 'ENOENT') {
            return missingValue
        }

        if (error instanceof z.ZodError) {
            throw new Error(`Invalid ${filePath}: ${error.message}`)
        }

        throw error
    }
}

export async function readJsonFile (filePath: string, displayPath = filePath): Promise<unknown> {
    return parseJsonText(await readFile(filePath, 'utf8'), displayPath)
}

export function parseJsonText (text: string, displayPath: string): unknown {
    try {
        return JSON.parse(text) as unknown
    } catch (error) {
        if (error instanceof SyntaxError) {
            throw new Error(`Invalid JSON in ${displayPath}: ${error.message}`)
        }

        throw error
    }
}

export async function writeJsonFile (filePath: string, value: unknown): Promise<void> {
    await atomicWriteFile(filePath, `${JSON.stringify(value, null, 2)}\n`)
}

export function isNodeError (error: unknown): error is NodeJS.ErrnoException {
    return error instanceof Error && 'code' in error
}

function toWriteFileAtomicData (contents: string|NodeJS.ArrayBufferView): string|Buffer {
    if (typeof contents === 'string' || Buffer.isBuffer(contents)) {
        return contents
    }

    return Buffer.from(contents.buffer, contents.byteOffset, contents.byteLength)
}

function isRetryableAtomicWriteError (error: unknown): boolean {
    return isNodeError(error) && typeof error.code === 'string' && RETRYABLE_ATOMIC_WRITE_CODES.has(error.code)
}

function delay (milliseconds: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, milliseconds))
}
