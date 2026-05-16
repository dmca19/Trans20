import { setImmediate } from 'node:timers/promises'

const DEFAULT_TIME_SLICE_MS = 2
const DEFAULT_CHECK_INTERVAL = 1

export type YieldControllerOptions = {
    clock?: () => number
    enabled?: boolean
    scheduler?: () => Promise<void>
    timeSliceMs?: number
    checkInterval?: number
}

export type YieldControllerStats = {
    yieldCount: number
}

export type YieldController = {
    forceYield: () => Promise<void>
    maybeYield: (index?: number) => Promise<void>
    stats: () => YieldControllerStats
}

export function yieldToEventLoop (): Promise<void> {
    return setImmediate()
}

export async function yieldEvery (index: number, interval = 100): Promise<void> {
    if (index > 0 && index % interval === 0) {
        await yieldToEventLoop()
    }
}

export function createYieldController (options: YieldControllerOptions = {}): YieldController {
    const enabled = options.enabled ?? !isEnvEnabled('TRANS20_YIELD_DISABLED')
    const timeSliceMs = normalizePositiveNumber(options.timeSliceMs, readPositiveNumberEnv('TRANS20_YIELD_TIME_SLICE_MS', DEFAULT_TIME_SLICE_MS))
    const checkInterval = normalizePositiveInteger(options.checkInterval, readPositiveNumberEnv('TRANS20_YIELD_CHECK_INTERVAL', DEFAULT_CHECK_INTERVAL))
    const clock = options.clock ?? (() => performance.now())
    const scheduler = options.scheduler ?? yieldToEventLoop
    let lastYieldTime = clock()
    let yieldCount = 0

    const forceYield = async (): Promise<void> => {
        await scheduler()
        lastYieldTime = clock()
        yieldCount += 1
    }

    return {
        forceYield,
        async maybeYield (index = checkInterval): Promise<void> {
            if (!enabled || index % checkInterval !== 0) {
                return
            }

            if (clock() - lastYieldTime >= timeSliceMs) {
                await forceYield()
            }
        },
        stats: () => ({ yieldCount }),
    }
}

function readPositiveNumberEnv (name: string, fallback: number): number {
    const value = process.env[name]

    if (!value) {
        return fallback
    }

    const parsed = Number(value)
    return normalizePositiveNumber(parsed, fallback)
}

function normalizePositiveNumber (value: number|undefined, fallback: number): number {
    return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback
}

function normalizePositiveInteger (value: number|undefined, fallback: number): number {
    return Math.max(1, Math.floor(normalizePositiveNumber(value, fallback)))
}

function isEnvEnabled (name: string): boolean {
    const value = process.env[name]

    return value === '1' || value?.toLowerCase() === 'true'
}
