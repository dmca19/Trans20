import { closeSync, fsyncSync, openSync, writeSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import path from 'node:path'

export type RunLogger = {
    filePath: string|null
    debugFilePath: string|null
    log: (type: string, payload: unknown) => void
    debug: (type: string, payload: unknown) => void
    flush: () => Promise<void>
    close: () => void
}

export type RunLoggerOptions = {
    enableRunLogs?: boolean
    enableDebugLogs?: boolean
}

export async function createRunLogger (
    root: string,
    logsDirectory = path.join(root, 'logs'),
    options: RunLoggerOptions = {},
): Promise<RunLogger> {
    const enableRunLogs = options.enableRunLogs ?? true
    const enableDebugLogs = options.enableDebugLogs ?? true

    const timestamp = formatLogTimestamp(new Date())
    const filePath = enableRunLogs ? path.join(logsDirectory, `trans20-${timestamp}-${process.pid}.jsonl`) : null
    const debugFilePath = enableDebugLogs ? path.join(logsDirectory, `trans20-debug-${timestamp}-${process.pid}.jsonl`) : null

    if (enableRunLogs || enableDebugLogs) {
        await mkdir(logsDirectory, { recursive: true })
    }

    const fd = filePath ? openSync(filePath, 'a') : null
    const debugFd = debugFilePath ? openSync(debugFilePath, 'a') : null
    let closed = false

    const logger: RunLogger = {
        filePath,
        debugFilePath,
        log: (type, payload) => {
            if (closed || fd === null) {
                return
            }

            writeLogEvent(fd, type, payload)
        },
        debug: (type, payload) => {
            if (closed || debugFd === null) {
                return
            }

            writeLogEvent(debugFd, type, payload)
        },
        flush: async () => {
            if (!closed) {
                if (fd !== null) {
                    fsyncSync(fd)
                }

                if (debugFd !== null) {
                    fsyncSync(debugFd)
                }
            }
        },
        close: () => {
            if (closed) {
                return
            }

            if (fd !== null) {
                fsyncSync(fd)
                closeSync(fd)
            }

            if (debugFd !== null) {
                fsyncSync(debugFd)
                closeSync(debugFd)
            }

            closed = true
        },
    }

    logger.log('run_started', {
        root,
        pid: process.pid,
        argv: process.argv,
        node: process.version,
    })
    logger.debug('debug_log_started', {
        root,
        pid: process.pid,
        runLog: filePath,
    })

    return logger
}

function writeLogEvent (fd: number, type: string, payload: unknown): void {
    const line = `${safeStringify({
        timestamp: new Date().toISOString(),
        type,
        payload,
    })}\n`

    writeLineSync(fd, line)
}

function writeLineSync (fd: number, line: string): void {
    writeSync(fd, line, undefined, 'utf8')
    fsyncSync(fd)
}

function formatLogTimestamp (date: Date): string {
    return date.toISOString().replace(/[:.]/g, '-')
}

function safeStringify (value: unknown): string {
    const seen = new WeakSet<object>()

    try {
        return JSON.stringify(value, (_key, childValue) => {
            if (typeof childValue === 'bigint') {
                return childValue.toString()
            }

            if (childValue instanceof Error) {
                return {
                    name: childValue.name,
                    message: childValue.message,
                    stack: childValue.stack,
                }
            }

            if (typeof childValue === 'object' && childValue !== null) {
                if (seen.has(childValue)) {
                    return '[Circular]'
                }

                seen.add(childValue)
            }

            return childValue
        })
    } catch (error) {
        return JSON.stringify({
            timestamp: new Date().toISOString(),
            type: 'logger_serialization_failed',
            payload: {
                error: error instanceof Error ? error.message : String(error),
            },
        })
    }
}
