import { atomicWriteFile } from './fileUtils.js'

const DEFAULT_FLUSH_INTERVAL_MS = 5_000
const unloadedState = Symbol('unloadedState')

export type PersistentStateStoreOptions<TState> = {
    resolveFilePath: (root: string) => string
    loadFromDisk: (filePath: string) => Promise<TState>
    flushIntervalMs?: number
    serialize?: (state: TState) => string
}

export type PersistentStateStore<TState> = {
    read: <T>(root: string, reader: (state: TState) => Promise<T>|T) => Promise<T>
    update: <T>(root: string, updater: (state: TState) => Promise<T>|T) => Promise<T>
    save: (root: string, state: TState) => Promise<void>
    flush: (root: string) => Promise<void>
    flushAll: () => Promise<void>
    close: (root: string) => Promise<void>
}

type StoreState<TState> = {
    filePath: string
    state: TState|typeof unloadedState
    statePromise: Promise<TState>|null
    operationQueue: Promise<void>
    flushTimer: NodeJS.Timeout|null
    flushPromise: Promise<void>|null
    memoryRevision: number
    dirtyRevision: number|null
    flushedRevision: number
    lastFlushError: unknown
}

type FlushSnapshot = {
    revision: number
    serializedState: string
}

export function createPersistentStateStore<TState> (
    options: PersistentStateStoreOptions<TState>,
): PersistentStateStore<TState> {
    const stores = new Map<string, StoreState<TState>>()
    const flushIntervalMs = options.flushIntervalMs ?? DEFAULT_FLUSH_INTERVAL_MS
    const serialize = options.serialize ?? defaultSerialize

    function getStore (root: string): StoreState<TState> {
        const filePath = options.resolveFilePath(root)
        const existingStore = stores.get(filePath)

        if (existingStore) {
            return existingStore
        }

        const store: StoreState<TState> = {
            filePath,
            state: unloadedState,
            statePromise: null,
            operationQueue: Promise.resolve(),
            flushTimer: null,
            flushPromise: null,
            memoryRevision: 0,
            dirtyRevision: null,
            flushedRevision: 0,
            lastFlushError: null,
        }

        stores.set(filePath, store)
        return store
    }

    function enqueueOperation<T> (
        store: StoreState<TState>,
        operation: () => Promise<T>|T,
    ): Promise<T> {
        const operationPromise = store.operationQueue.then(operation)

        store.operationQueue = operationPromise.then(() => undefined, () => undefined)
        return operationPromise
    }

    async function loadStoreState (store: StoreState<TState>): Promise<TState> {
        if (store.state !== unloadedState) {
            return store.state
        }

        store.statePromise ??= options.loadFromDisk(store.filePath)
        store.state = await store.statePromise
        return store.state
    }

    function markStoreDirty (store: StoreState<TState>): void {
        store.memoryRevision += 1
        store.dirtyRevision = store.memoryRevision
        scheduleFlush(store)
    }

    function scheduleFlush (store: StoreState<TState>): void {
        if (store.flushTimer) {
            return
        }

        store.flushTimer = setTimeout(() => {
            store.flushTimer = null
            void flushStore(store, false).catch(error => {
                store.lastFlushError = error
                if (store.dirtyRevision !== null) {
                    scheduleFlush(store)
                }
            })
        }, flushIntervalMs)
        store.flushTimer.unref?.()
    }

    function clearFlushTimer (store: StoreState<TState>): void {
        if (!store.flushTimer) {
            return
        }

        clearTimeout(store.flushTimer)
        store.flushTimer = null
    }

    async function flushStore (store: StoreState<TState>, force: boolean): Promise<void> {
        if (store.flushPromise) {
            await store.flushPromise
        }

        const snapshot = await captureFlushSnapshot(store, force)

        if (!snapshot) {
            return
        }

        const flushPromise = atomicWriteFile(store.filePath, snapshot.serializedState)

        store.flushPromise = flushPromise

        try {
            await flushPromise
            await enqueueOperation(store, () => {
                store.flushedRevision = Math.max(store.flushedRevision, snapshot.revision)
                store.lastFlushError = null

                if (store.memoryRevision === snapshot.revision && store.dirtyRevision !== null && store.dirtyRevision <= snapshot.revision) {
                    store.dirtyRevision = null
                }
            })
        } finally {
            if (store.flushPromise === flushPromise) {
                store.flushPromise = null
            }
        }

        if (store.dirtyRevision !== null) {
            scheduleFlush(store)
        }
    }

    async function captureFlushSnapshot (store: StoreState<TState>, force: boolean): Promise<FlushSnapshot|null> {
        return enqueueOperation(store, async () => {
            if (!force && store.dirtyRevision === null) {
                return null
            }

            if (store.state === unloadedState) {
                if (!force) {
                    return null
                }

                await loadStoreState(store)
            }

            if (store.state === unloadedState || (!force && store.dirtyRevision === null)) {
                return null
            }

            clearFlushTimer(store)

            return {
                revision: store.memoryRevision,
                serializedState: serialize(store.state),
            }
        })
    }

    return {
        read: (root, reader) => {
            const store = getStore(root)
            return enqueueOperation(store, async () => reader(await loadStoreState(store)))
        },
        update: (root, updater) => {
            const store = getStore(root)
            return enqueueOperation(store, async () => {
                const state = await loadStoreState(store)
                const result = await updater(state)

                markStoreDirty(store)
                return result
            })
        },
        save: async (root, state) => {
            const store = getStore(root)

            await enqueueOperation(store, () => {
                store.state = state
                store.statePromise = Promise.resolve(state)
                markStoreDirty(store)
            })
            await flushStore(store, true)
        },
        flush: async root => {
            const store = getStore(root)

            await store.operationQueue
            await flushStore(store, true)
        },
        flushAll: async () => {
            await Promise.all(Array.from(stores.values(), async store => {
                await store.operationQueue
                await flushStore(store, true)
            }))
        },
        close: async root => {
            const filePath = options.resolveFilePath(root)
            const store = stores.get(filePath)

            if (!store) {
                return
            }

            await store.operationQueue
            await flushStore(store, true)
            clearFlushTimer(store)
            stores.delete(filePath)
        },
    }
}

function defaultSerialize<TState> (state: TState): string {
    return `${JSON.stringify(state, null, 2)}\n`
}
