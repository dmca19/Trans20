import { randomUUID } from 'node:crypto'

import { yieldToEventLoop } from './eventLoop.js'

export type ToolCallEvent = {
    id: string
    toolName: string
    status: 'started'|'completed'|'failed'
    input: string
    output?: string
    error?: string
}

export type ToolCallLogger = (event: ToolCallEvent) => void

export async function withToolLogging (
    toolName: string,
    input: unknown,
    onToolEvent: ToolCallLogger,
    run: () => Promise<string>,
): Promise<string> {
    const id = randomUUID()
    const serializedInput = stringifyForLog(input, Number.MAX_SAFE_INTEGER)

    onToolEvent({
        id,
        toolName,
        status: 'started',
        input: serializedInput,
    })
    await yieldToEventLoop()

    try {
        const output = await run()
        onToolEvent({
            id,
            toolName,
            status: 'completed',
            input: serializedInput,
            output,
        })
        await yieldToEventLoop()
        return output
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        onToolEvent({
            id,
            toolName,
            status: 'failed',
            input: serializedInput,
            error: message,
        })
        await yieldToEventLoop()
        throw error
    }
}

export function jsonOutput (value: unknown): string {
    return JSON.stringify(value)
}

export function stringifyForLog (value: unknown, maxLength: number): string {
    try {
        return truncate(JSON.stringify(value, null, 2), maxLength)
    } catch {
        return String(value)
    }
}

export function truncate (value: string, maxLength: number): string {
    if (value.length <= maxLength) {
        return value
    }

    return `${value.slice(0, maxLength)}\n... truncated ...`
}
