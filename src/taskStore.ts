import { createHash, randomInt } from 'node:crypto'
import { readdir, readFile, stat } from 'node:fs/promises'
import path from 'node:path'

import { z } from 'zod'

import type { AppMode } from './App.js'
import { atomicWriteFile, isNodeError, readJsonFile } from './fileUtils.js'
import { partialTaskConfigSnapshotSchema, taskConfigSnapshotSchema, type AgentConfig } from './config.js'
import { GLOSSARY_FILE_PATH } from './glossaryStore.js'
import { normalizePathId, resolveProjectFile, toPosixPath } from './pathUtils.js'
import { TRANSLATED_MANUAL_TRANS_FILE_PATH, TRANSLATION_STATE_FILE_PATH } from './translationStore.js'

export const OUTPUT_DIRECTORY_PATH = 'output'
export const TASK_FILE_NAME = 'task.json'
export const LOGS_DIRECTORY_NAME = 'logs'
export const SNAPSHOTS_DIRECTORY_NAME = 'snapshots'

export type TaskStatus = 'running'|'interrupted'|'completed'|'failed'
// Differs from AppStage: TaskStage is persisted task.json progress.
export type TaskStage = 'created'|'initializing'|'preflighting'|'extracting'|'reviewing'|'exporting'|'ready'
export type TaskBatchStatus = 'pending'|'running'|'completed'|'failed'|'skipped'

export type TaskBatchTransaction = {
    kind: 'glossary_worker'
    status: 'running'
    snapshot_path: string
    started_at: string
}

export type TaskBatchProgress = {
    batch_id: string
    batch_number: number
    batch_start_index: number
    batch_end_index: number
    status: TaskBatchStatus
    slot_index?: number
    lifecycle?: number
    started_at?: string
    completed_at?: string
    failed_at?: string
    error?: string
    transaction?: TaskBatchTransaction
}

export type TaskReviewTransaction = {
    kind: 'glossary_review'
    status: 'running'
    snapshot_path: string
    started_at: string
    review_window_id?: string
    completed_batch_numbers: number[]
}

export type TaskProgress = {
    worker_lifecycle_mode?: 'wave'
    total_batches: number
    completed_batches: number
    failed_batches: number
    skipped_batches: number
    batches: TaskBatchProgress[]
    completed_reviews?: number
    failed_reviews?: number
    exported_path?: string
    review_transaction?: TaskReviewTransaction
}

export type TaskConfigSnapshot = Partial<Omit<AgentConfig, 'url'|'key'|'enableRunLogs'|'enableDebugLogs'>>

export type TaskConfigDifference = {
    field: keyof TaskConfigSnapshot
    currentValue: unknown
    taskValue: unknown
}

export type TaskGlossarySource = {
    task_code: string
    glossary_path: string
    manual_trans_file_hash: string
    selected_at: string
}

export type ProjectTask = {
    version: 1
    task_code: string
    mode: AppMode
    status: TaskStatus
    stage: TaskStage
    manual_trans_file: string
    manual_trans_file_hash: string|null
    manual_trans_file_size_bytes: number|null
    source_file_id: string|null
    created_at: string
    updated_at: string
    completed_at?: string
    last_error?: string
    log_files: {
        run_log?: string|null
        debug_log?: string|null
    }
    glossary_source?: TaskGlossarySource|null
    config_snapshot: TaskConfigSnapshot
    progress: TaskProgress
}

export type CreateTaskInput = {
    mode: AppMode
    config: AgentConfig
}

export type TaskListOptions = {
    mode?: AppMode
    manualTransFile?: string
    resumableOnly?: boolean
}

export type TaskUpdate = Partial<Pick<ProjectTask, 'stage'|'status'|'source_file_id'>> & {
    log_files?: Partial<ProjectTask['log_files']>
    glossary_source?: TaskGlossarySource|null
    progress?: Partial<Omit<TaskProgress, 'review_transaction'>> & {
        review_transaction?: TaskReviewTransaction|null
    }
    completed_at?: string|null
    last_error?: string|null
}

const optionalStringSchema = z.string().optional().catch(undefined)
const optionalIntegerSchema = z.number().int().optional().catch(undefined)
const optionalNullableStringSchema = z.string().nullable().optional().catch(null)
const optionalNullableIntegerSchema = z.number().int().nullable().optional().catch(null)
const taskCodeSchema = z.string().regex(/^[a-z]+-[a-z]+$/)
const taskModeSchema = z.enum(['glossary', 'translation'])
const taskStatusSchema = z.enum(['running', 'interrupted', 'completed', 'failed'])
const taskStageSchema = z.enum(['created', 'initializing', 'preflighting', 'extracting', 'reviewing', 'exporting', 'ready'])
const taskBatchStatusSchema = z.enum(['pending', 'running', 'completed', 'failed', 'skipped'])

const taskBatchTransactionSchema = z.object({
    kind: z.literal('glossary_worker'),
    status: z.literal('running'),
    snapshot_path: z.string(),
    started_at: z.string(),
})

const taskReviewTransactionSchema = z.object({
    kind: z.literal('glossary_review'),
    status: z.literal('running'),
    snapshot_path: z.string(),
    started_at: z.string(),
    review_window_id: optionalStringSchema,
    completed_batch_numbers: z.array(z.number().int()),
})

const taskBatchProgressSchema = z.object({
    batch_id: z.string(),
    batch_number: z.number().int(),
    batch_start_index: z.number().int(),
    batch_end_index: z.number().int(),
    status: taskBatchStatusSchema,
    slot_index: optionalIntegerSchema,
    lifecycle: optionalIntegerSchema,
    started_at: optionalStringSchema,
    completed_at: optionalStringSchema,
    failed_at: optionalStringSchema,
    error: optionalStringSchema,
    transaction: taskBatchTransactionSchema.optional(),
})

const taskProgressInputSchema = z.object({
    worker_lifecycle_mode: z.literal('wave').optional().catch(undefined),
    total_batches: optionalIntegerSchema,
    completed_batches: optionalIntegerSchema,
    failed_batches: optionalIntegerSchema,
    skipped_batches: optionalIntegerSchema,
    batches: z.array(z.unknown()).optional().catch(undefined),
    completed_reviews: optionalIntegerSchema,
    failed_reviews: optionalIntegerSchema,
    exported_path: optionalStringSchema,
    review_transaction: z.unknown().optional(),
})

const taskLogFilesSchema = z.object({
    run_log: optionalStringSchema,
    debug_log: optionalStringSchema,
}).optional().catch({})

const taskGlossarySourceSchema = z.object({
    task_code: taskCodeSchema,
    glossary_path: z.string().min(1),
    manual_trans_file_hash: z.string().min(1),
    selected_at: z.string().min(1),
}).nullable().optional().catch(null)

const projectTaskInputSchema = z.object({
    task_code: taskCodeSchema,
    mode: taskModeSchema,
    status: taskStatusSchema,
    stage: taskStageSchema,
    manual_trans_file: z.string().min(1),
    manual_trans_file_hash: optionalNullableStringSchema,
    manual_trans_file_size_bytes: optionalNullableIntegerSchema,
    source_file_id: optionalStringSchema,
    created_at: z.string().min(1),
    updated_at: z.string().min(1),
    completed_at: optionalStringSchema,
    last_error: optionalStringSchema,
    log_files: taskLogFilesSchema,
    glossary_source: taskGlossarySourceSchema,
    config_snapshot: z.unknown().optional(),
    progress: z.unknown().optional(),
})

const adjectives = [
    'able',
    'brisk',
    'calm',
    'clever',
    'diligent',
    'direct',
    'eager',
    'exact',
    'fair',
    'gentle',
    'honest',
    'keen',
    'lucid',
    'nimble',
    'patient',
    'plain',
    'quick',
    'steady',
    'tidy',
    'vivid',
]

const nouns = [
    'amber',
    'brook',
    'cedar',
    'clover',
    'delta',
    'ember',
    'field',
    'fig',
    'harbor',
    'iris',
    'juniper',
    'maple',
    'meadow',
    'orchid',
    'quartz',
    'river',
    'sage',
    'spruce',
    'violet',
    'willow',
]

const generatedTaskCodes = new Set<string>()
const taskOperationQueues = new Map<string, Promise<void>>()

const taskConfigSnapshotFields = Object.keys(taskConfigSnapshotSchema.shape) as Array<keyof TaskConfigSnapshot>

export async function createTask (root: string, input: CreateTaskInput): Promise<ProjectTask> {
    const now = new Date().toISOString()
    const manualTransFileFingerprint = await hashProjectFile(root, input.config.manualTransFile)
    const task: ProjectTask = {
        version: 1,
        task_code: await generateTaskCode(root),
        mode: input.mode,
        status: 'running',
        stage: 'created',
        manual_trans_file: input.config.manualTransFile,
        manual_trans_file_hash: manualTransFileFingerprint.hash,
        manual_trans_file_size_bytes: manualTransFileFingerprint.sizeBytes,
        source_file_id: manualTransFileFingerprint.relativePath,
        created_at: now,
        updated_at: now,
        log_files: {},
        config_snapshot: createTaskConfigSnapshot(input.config),
        progress: createEmptyProgress(),
    }

    await writeTask(root, task)
    return task
}

export function createTaskConfigSnapshot (config: AgentConfig): TaskConfigSnapshot {
    return taskConfigSnapshotSchema.parse(Object.fromEntries(
        taskConfigSnapshotFields.map(field => [field, config[field]]),
    ))
}

export function isCompleteTaskConfigSnapshot (snapshot: TaskConfigSnapshot): boolean {
    return taskConfigSnapshotSchema.safeParse(snapshot).success
}

export function compareTaskConfigSnapshot (
    currentConfig: AgentConfig,
    snapshot: TaskConfigSnapshot,
): TaskConfigDifference[] {
    const snapshotResult = partialTaskConfigSnapshotSchema.safeParse(snapshot)
    const parsedSnapshot = snapshotResult.success ? snapshotResult.data : {}

    return taskConfigSnapshotFields
        .filter(field => parsedSnapshot[field] !== undefined)
        .filter(field => field !== 'manualTransFile')
        .filter(field => !Object.is(currentConfig[field], parsedSnapshot[field]))
        .map(field => ({
            field,
            currentValue: currentConfig[field],
            taskValue: parsedSnapshot[field],
        }))
}

export function applyTaskConfigSnapshot (currentConfig: AgentConfig, snapshot: TaskConfigSnapshot): AgentConfig {
    const snapshotResult = taskConfigSnapshotSchema.safeParse(snapshot)

    if (!snapshotResult.success) {
        throw new Error('Task config snapshot is incomplete and cannot be used to resume.')
    }

    return {
        ...currentConfig,
        ...snapshotResult.data,
        manualTransFile: currentConfig.manualTransFile,
    }
}

export async function listTasks (root: string, options: TaskListOptions = {}): Promise<ProjectTask[]> {
    const directory = path.join(root, OUTPUT_DIRECTORY_PATH)
    let entries: import('node:fs').Dirent[]

    try {
        entries = await readdir(directory, { withFileTypes: true })
    } catch (error) {
        if (isNodeError(error) && error.code === 'ENOENT') {
            return []
        }

        throw error
    }

    const manualTransFileHash = options.manualTransFile
        ? (await hashProjectFile(root, options.manualTransFile)).hash
        : null
    const tasks = await Promise.all(entries
        .filter(entry => entry.isDirectory())
        .map(async entry => {
            try {
                return parseTask(await readJsonFile(
                    path.join(directory, entry.name, TASK_FILE_NAME),
                    getTaskFileRelativePath(entry.name, TASK_FILE_NAME),
                ))
            } catch {
                return null
            }
        }))

    const filteredTasks = tasks
        .filter((task): task is ProjectTask => task !== null)
        .filter(task => options.mode ? task.mode === options.mode : true)
        .filter(task => manualTransFileHash ? task.manual_trans_file_hash === manualTransFileHash : true)
        .filter(task => options.resumableOnly ? task.status !== 'completed' : true)

    if (options.resumableOnly) {
        const validationResults = await Promise.all(filteredTasks.map(async task => {
            try {
                await validateTaskManualTransFile(root, task, options.manualTransFile)
                await validateTaskRuntimeStateFile(root, task)
                return task
            } catch {
                return null
            }
        }))

        return validationResults
            .filter((task): task is ProjectTask => task !== null)
            .sort((left, right) => right.updated_at.localeCompare(left.updated_at) || right.created_at.localeCompare(left.created_at))
    }

    return filteredTasks
        .sort((left, right) => right.updated_at.localeCompare(left.updated_at) || right.created_at.localeCompare(left.created_at))
}

export async function listGlossarySourceTasks (root: string, manualTransFile: string): Promise<ProjectTask[]> {
    const tasks = await listTasks(root, {
        mode: 'glossary',
        manualTransFile,
    })
    const candidates = await Promise.all(tasks.map(async task => {
        if (task.status !== 'completed' || task.manual_trans_file_hash === null) {
            return null
        }

        try {
            await validateTaskManualTransFile(root, task, manualTransFile)
            await validateTaskRuntimeStateFile(root, task)
            return task
        } catch {
            return null
        }
    }))

    return candidates
        .filter((task): task is ProjectTask => task !== null)
        .sort((left, right) => right.updated_at.localeCompare(left.updated_at) || right.created_at.localeCompare(left.created_at))
}

export async function validateGlossarySource (
    root: string,
    source: TaskGlossarySource,
    manualTransFile: string,
): Promise<void> {
    const task = await loadTask(root, source.task_code)

    if (task.mode !== 'glossary') {
        throw new Error(`Selected glossary source ${source.task_code} is not a glossary task.`)
    }

    if (task.status !== 'completed') {
        throw new Error(`Selected glossary source ${source.task_code} is not completed.`)
    }

    await validateTaskManualTransFile(root, task, manualTransFile)

    if (task.manual_trans_file_hash !== source.manual_trans_file_hash) {
        throw new Error(`Selected glossary source ${source.task_code} hash does not match the recorded glossary source hash.`)
    }

    const expectedGlossaryPath = getTaskFileRelativePath(source.task_code, GLOSSARY_FILE_PATH)

    if (normalizePathId(source.glossary_path) !== normalizePathId(expectedGlossaryPath)) {
        throw new Error(`Selected glossary source ${source.task_code} points to ${source.glossary_path}, expected ${expectedGlossaryPath}.`)
    }

    await validateTaskRuntimeStateFile(root, task)
}

export async function loadTask (root: string, taskCode: string): Promise<ProjectTask> {
    const queue = taskOperationQueues.get(resolveTaskPath(root, taskCode))
    if (queue) {
        await queue
    }

    return readTaskFromDisk(root, taskCode)
}

async function readTaskFromDisk (root: string, taskCode: string): Promise<ProjectTask> {
    return parseTask(await readJsonFile(resolveTaskPath(root, taskCode), getTaskFileRelativePath(taskCode, TASK_FILE_NAME)))
}

function applyTaskUpdate (task: ProjectTask, update: TaskUpdate): ProjectTask {
    const now = new Date().toISOString()
    const nextGlossarySource = 'glossary_source' in update
        ? { glossary_source: update.glossary_source ?? null }
        : ('glossary_source' in task ? { glossary_source: task.glossary_source } : {})
    const nextLogFiles = {
        ...task.log_files,
        ...update.log_files,
    }

    if (nextLogFiles.run_log === null) {
        delete nextLogFiles.run_log
    }

    if (nextLogFiles.debug_log === null) {
        delete nextLogFiles.debug_log
    }

    const nextTask: ProjectTask = {
        ...task,
        ...('stage' in update ? { stage: update.stage ?? task.stage } : {}),
        ...('status' in update ? { status: update.status ?? task.status } : {}),
        ...('source_file_id' in update ? { source_file_id: update.source_file_id ?? null } : {}),
        ...(typeof update.last_error === 'string' ? { last_error: update.last_error } : {}),
        log_files: nextLogFiles,
        ...nextGlossarySource,
        progress: applyProgressUpdate(task.progress, update.progress),
        updated_at: now,
    }

    if (update.completed_at === null) {
        delete nextTask.completed_at
    } else if (update.completed_at !== undefined) {
        nextTask.completed_at = update.completed_at
    }

    if (update.last_error === null) {
        delete nextTask.last_error
    }

    return nextTask
}

function applyProgressUpdate (
    progress: TaskProgress,
    update: TaskUpdate['progress'],
): TaskProgress {
    if (!update) {
        return progress
    }

    const { review_transaction: reviewTransactionUpdate, ...restUpdate } = update
    const nextProgress: TaskProgress = {
        ...progress,
        ...restUpdate,
    }

    if (reviewTransactionUpdate === null) {
        delete nextProgress.review_transaction
    } else if (reviewTransactionUpdate !== undefined) {
        nextProgress.review_transaction = reviewTransactionUpdate
    }

    return nextProgress
}

export async function updateTask (root: string, taskCode: string, update: TaskUpdate): Promise<ProjectTask> {
    return enqueueTaskOperation(root, taskCode, async () => {
        const task = await readTaskFromDisk(root, taskCode)
        const nextTask = applyTaskUpdate(task, update)

        await writeTask(root, nextTask)
        return nextTask
    })
}

export async function markTaskInterrupted (root: string, taskCode: string, manualTransFile?: string): Promise<ProjectTask> {
    const task = await loadTask(root, taskCode)

    if (task.status === 'completed') {
        return task
    }

    await validateTaskManualTransFile(root, task, manualTransFile)
    await validateTaskRuntimeStateFile(root, task)

    return updateTask(root, taskCode, {
        status: 'interrupted',
    })
}

export async function updateTaskBatch (
    root: string,
    taskCode: string,
    batch: Omit<TaskBatchProgress, 'status'> & { status: TaskBatchStatus },
): Promise<ProjectTask> {
    return enqueueTaskOperation(root, taskCode, async () => {
        const task = await readTaskFromDisk(root, taskCode)
        const existingBatch = task.progress.batches.find(item => item.batch_id === batch.batch_id)
        const effectiveBatch = shouldPreserveExistingBatchStatus(existingBatch, batch.status)
            ? {
                ...existingBatch,
                batch_number: batch.batch_number,
                batch_start_index: batch.batch_start_index,
                batch_end_index: batch.batch_end_index,
            }
            : batch
        const nextBatch: TaskBatchProgress = {
            ...effectiveBatch,
            ...getPreservedBatchLifecycleMetadata(existingBatch, effectiveBatch),
            ...(effectiveBatch.status === 'running' && effectiveBatch.transaction === undefined && existingBatch?.transaction
                ? { transaction: existingBatch.transaction }
                : {}),
        }
        const batches = task.progress.batches.filter(item => item.batch_id !== batch.batch_id)
        batches.push(nextBatch)
        batches.sort((left, right) => left.batch_number - right.batch_number)

        const completedBatches = batches.filter(item => item.status === 'completed').length
        const failedBatches = batches.filter(item => item.status === 'failed').length
        const skippedBatches = batches.filter(item => item.status === 'skipped').length
        const nextTask = applyTaskUpdate(task, {
            progress: {
                ...task.progress,
                total_batches: Math.max(task.progress.total_batches, batches.length),
                completed_batches: completedBatches,
                failed_batches: failedBatches,
                skipped_batches: skippedBatches,
                batches,
            },
        })

        await writeTask(root, nextTask)
        return nextTask
    })
}

function shouldPreserveExistingBatchStatus (
    existingBatch: TaskBatchProgress|undefined,
    nextStatus: TaskBatchStatus,
): existingBatch is TaskBatchProgress {
    if (!existingBatch) {
        return false
    }

    return existingBatch.status === 'completed' && nextStatus === 'skipped'
}

function getPreservedBatchLifecycleMetadata (
    existingBatch: TaskBatchProgress|undefined,
    nextBatch: TaskBatchProgress,
): Pick<TaskBatchProgress, 'slot_index'|'lifecycle'> {
    if (!existingBatch || nextBatch.status !== 'completed') {
        return {}
    }

    return {
        ...(nextBatch.slot_index === undefined && existingBatch.slot_index !== undefined ? { slot_index: existingBatch.slot_index } : {}),
        ...(nextBatch.lifecycle === undefined && existingBatch.lifecycle !== undefined ? { lifecycle: existingBatch.lifecycle } : {}),
    }
}

export async function replaceTaskProgress (root: string, taskCode: string, progress: TaskProgress): Promise<ProjectTask> {
    return updateTask(root, taskCode, { progress })
}

export function describeTaskForSelection (task: ProjectTask): string {
    const progress = task.progress.total_batches > 0
        ? `${task.progress.completed_batches}/${task.progress.total_batches}`
        : '-'
    const error = task.last_error ? ` error=${truncateSingleLine(task.last_error, 48)}` : ''

    return `${task.task_code} ${task.mode} ${task.stage} ${task.status} batches=${progress} updated=${task.updated_at}${error}`
}

export function createEmptyProgress (): TaskProgress {
    return {
        worker_lifecycle_mode: 'wave',
        total_batches: 0,
        completed_batches: 0,
        failed_batches: 0,
        skipped_batches: 0,
        batches: [],
    }
}

export function createTaskRunPaths (root: string, taskCode: string): {
    taskDirectory: string
    taskDirectoryRelativePath: string
    logDirectory: string
    snapshotDirectory: string
    glossaryFile: string
    translationStateFile: string
    translatedManualTransFile: string
} {
    return {
        taskDirectory: resolveTaskDirectoryPath(root, taskCode),
        taskDirectoryRelativePath: getTaskDirectoryRelativePath(taskCode),
        logDirectory: resolveTaskLogDirectoryPath(root, taskCode),
        snapshotDirectory: resolveTaskSnapshotDirectoryPath(root, taskCode),
        glossaryFile: resolveTaskFilePath(root, taskCode, GLOSSARY_FILE_PATH),
        translationStateFile: resolveTaskFilePath(root, taskCode, TRANSLATION_STATE_FILE_PATH),
        translatedManualTransFile: resolveTaskFilePath(root, taskCode, TRANSLATED_MANUAL_TRANS_FILE_PATH),
    }
}

export function getTaskDirectoryRelativePath (taskCode: string): string {
    validateTaskCode(taskCode)
    return toPosixPath(path.join(OUTPUT_DIRECTORY_PATH, taskCode))
}

export function getTaskFileRelativePath (taskCode: string, fileName: string): string {
    validateTaskCode(taskCode)
    return toPosixPath(path.join(OUTPUT_DIRECTORY_PATH, taskCode, fileName))
}

export function resolveTaskDirectoryPath (root: string, taskCode: string): string {
    validateTaskCode(taskCode)
    return path.join(root, OUTPUT_DIRECTORY_PATH, taskCode)
}

export function resolveTaskLogDirectoryPath (root: string, taskCode: string): string {
    return path.join(resolveTaskDirectoryPath(root, taskCode), LOGS_DIRECTORY_NAME)
}

export function resolveTaskSnapshotDirectoryPath (root: string, taskCode: string): string {
    return path.join(resolveTaskDirectoryPath(root, taskCode), SNAPSHOTS_DIRECTORY_NAME)
}

export function resolveTaskFilePath (root: string, taskCode: string, fileName: string): string {
    return path.join(resolveTaskDirectoryPath(root, taskCode), fileName)
}

export function createGlossarySourceFromTask (task: ProjectTask): TaskGlossarySource {
    if (task.mode !== 'glossary') {
        throw new Error(`Task ${task.task_code} is not a glossary task.`)
    }

    if (!task.manual_trans_file_hash) {
        throw new Error(`Task ${task.task_code} does not have a ManualTransFile hash.`)
    }

    return {
        task_code: task.task_code,
        glossary_path: getTaskFileRelativePath(task.task_code, GLOSSARY_FILE_PATH),
        manual_trans_file_hash: task.manual_trans_file_hash,
        selected_at: new Date().toISOString(),
    }
}

async function generateTaskCode (root: string): Promise<string> {
    const existingCodes = new Set((await listTasks(root)).map(task => task.task_code))

    for (let attempt = 0; attempt < 200; attempt += 1) {
        const code = `${adjectives[randomInt(adjectives.length)]}-${nouns[randomInt(nouns.length)]}`

        if (!existingCodes.has(code) && !generatedTaskCodes.has(code)) {
            generatedTaskCodes.add(code)
            return code
        }
    }

    throw new Error('Unable to generate a unique task code.')
}

async function writeTask (root: string, task: ProjectTask): Promise<void> {
    const filePath = resolveTaskPath(root, task.task_code)
    await atomicWriteFile(filePath, `${JSON.stringify(task, null, 2)}\n`)
}

function enqueueTaskOperation<T> (
    root: string,
    taskCode: string,
    operation: () => Promise<T>,
): Promise<T> {
    const filePath = resolveTaskPath(root, taskCode)
    const previousQueue = taskOperationQueues.get(filePath) ?? Promise.resolve()
    const operationPromise = previousQueue.then(operation, operation)
    const nextQueue = operationPromise.then(() => undefined, () => undefined)

    taskOperationQueues.set(filePath, nextQueue)
    void nextQueue.finally(() => {
        if (taskOperationQueues.get(filePath) === nextQueue) {
            taskOperationQueues.delete(filePath)
        }
    })

    return operationPromise
}

function resolveTaskPath (root: string, taskCode: string): string {
    return path.join(resolveTaskDirectoryPath(root, taskCode), TASK_FILE_NAME)
}

function parseTask (value: unknown): ProjectTask {
    if (!isRecord(value)) {
        throw new Error('Task file must contain a JSON object.')
    }

    const taskCode = typeof value.task_code === 'string' ? value.task_code : null

    if (!taskCode || !taskCodeSchema.safeParse(taskCode).success) {
        throw new Error('Task file has invalid task_code.')
    }

    const parsedTask = projectTaskInputSchema.safeParse(value)

    if (parsedTask.success) {
        const task = parsedTask.data

        return {
            version: 1,
            task_code: task.task_code,
            mode: task.mode,
            status: task.status,
            stage: task.stage,
            manual_trans_file: task.manual_trans_file,
            manual_trans_file_hash: task.manual_trans_file_hash ?? null,
            manual_trans_file_size_bytes: task.manual_trans_file_size_bytes ?? null,
            source_file_id: task.source_file_id ?? null,
            created_at: task.created_at,
            updated_at: task.updated_at,
            ...(task.completed_at !== undefined ? { completed_at: task.completed_at } : {}),
            ...(task.last_error !== undefined ? { last_error: task.last_error } : {}),
            log_files: task.log_files ?? {},
            ...(task.glossary_source !== undefined ? { glossary_source: task.glossary_source } : {}),
            config_snapshot: parseConfigSnapshot(task.config_snapshot),
            progress: parseProgress(task.progress),
        }
    }

    if (!taskModeSchema.safeParse(value.mode).success) {
        throw new Error(`Task ${taskCode} has invalid mode.`)
    }

    if (!taskStatusSchema.safeParse(value.status).success
        || !taskStageSchema.safeParse(value.stage).success
        || typeof value.manual_trans_file !== 'string'
        || value.manual_trans_file.length === 0
        || typeof value.created_at !== 'string'
        || value.created_at.length === 0
        || typeof value.updated_at !== 'string'
        || value.updated_at.length === 0) {
        throw new Error(`Task ${taskCode} is missing required fields.`)
    }

    throw new Error(`Task ${taskCode} is invalid.`)
}

export async function validateTaskManualTransFile (root: string, task: ProjectTask, manualTransFile = task.manual_trans_file): Promise<void> {
    if (!task.manual_trans_file_hash) {
        throw new Error(`Task ${task.task_code} cannot be resumed because it was created before ManualTransFile hashing was recorded.`)
    }

    const fingerprint = await hashProjectFile(root, manualTransFile)

    if (task.manual_trans_file_hash !== fingerprint.hash) {
        throw new Error(`Task ${task.task_code} ManualTransFile hash changed from ${task.manual_trans_file_hash} to ${fingerprint.hash}.`)
    }
}

export async function validateTaskRuntimeStateFile (root: string, task: ProjectTask): Promise<void> {
    const runtimeStatePath = getTaskRuntimeStatePath(task)

    try {
        const runtimeStateStat = await stat(path.resolve(root, runtimeStatePath))

        if (!runtimeStateStat.isFile()) {
            throw new Error(`Task ${task.task_code} cannot be resumed because ${runtimeStatePath} is not a file.`)
        }
    } catch (error) {
        if (isNodeError(error) && error.code === 'ENOENT') {
            throw new Error(`Task ${task.task_code} cannot be resumed because ${runtimeStatePath} is missing.`)
        }

        throw error
    }
}

function getTaskRuntimeStatePath (task: ProjectTask): string {
    return task.mode === 'translation'
        ? getTaskFileRelativePath(task.task_code, TRANSLATION_STATE_FILE_PATH)
        : getTaskFileRelativePath(task.task_code, GLOSSARY_FILE_PATH)
}

async function hashProjectFile (root: string, requestedPath: string): Promise<{
    relativePath: string
    hash: string
    sizeBytes: number
}> {
    const file = await resolveProjectFile(root, requestedPath)
    const bytes = await readFile(file.realFilePath)

    return {
        relativePath: file.relativePath,
        hash: `sha256:${createHash('sha256').update(bytes).digest('hex')}`,
        sizeBytes: bytes.byteLength,
    }
}

function parseConfigSnapshot (value: unknown): TaskConfigSnapshot {
    const result = partialTaskConfigSnapshotSchema.safeParse(value)
    return result.success ? result.data : {}
}

function parseProgress (value: unknown): TaskProgress {
    const result = taskProgressInputSchema.safeParse(value)

    if (!result.success) {
        return createEmptyProgress()
    }

    const progress = result.data
    const batches: TaskBatchProgress[] = []

    for (const batch of progress.batches ?? []) {
        const parsedBatch = taskBatchProgressSchema.safeParse(batch)

        if (parsedBatch.success) {
            batches.push(parsedBatch.data)
        }
    }

    const reviewTransaction = taskReviewTransactionSchema.safeParse(progress.review_transaction)

    return {
        ...(progress.worker_lifecycle_mode !== undefined ? { worker_lifecycle_mode: progress.worker_lifecycle_mode } : {}),
        total_batches: progress.total_batches ?? batches.length,
        completed_batches: progress.completed_batches ?? batches.filter(item => item.status === 'completed').length,
        failed_batches: progress.failed_batches ?? batches.filter(item => item.status === 'failed').length,
        skipped_batches: progress.skipped_batches ?? batches.filter(item => item.status === 'skipped').length,
        batches,
        ...(progress.completed_reviews !== undefined ? { completed_reviews: progress.completed_reviews } : {}),
        ...(progress.failed_reviews !== undefined ? { failed_reviews: progress.failed_reviews } : {}),
        ...(progress.exported_path !== undefined ? { exported_path: progress.exported_path } : {}),
        ...(reviewTransaction.success ? { review_transaction: reviewTransaction.data } : {}),
    }
}

function validateTaskCode (taskCode: string): void {
    if (!/^[a-z]+-[a-z]+$/.test(taskCode)) {
        throw new Error(`Invalid task code: ${taskCode}`)
    }
}

// Differs from translationTools.truncateSingleLine: folds whitespace before truncating.
function truncateSingleLine (value: string, maxLength: number): string {
    const singleLine = value.replace(/\s+/g, ' ').trim()
    return singleLine.length > maxLength ? `${singleLine.slice(0, maxLength - 3)}...` : singleLine
}

// Same plain-object guard as agent/glossaryStore/glossaryTools/translationStore isRecord.
function isRecord (value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value)
}
