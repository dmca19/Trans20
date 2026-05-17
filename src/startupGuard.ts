import type { AppMode } from './App.js'

// Dead code for now: replaced by output/<task_code>, kept for possible future reuse.
export type CompletedResultFileBlocker = {
    mode: AppMode
    resultPaths: string[]
}

export async function findCompletedResultFileBlocker (
    _root: string,
    _mode: AppMode,
    _manualTransFile: string,
): Promise<CompletedResultFileBlocker|null> {
    return null
}

export function formatCompletedResultFileWarning (blocker: CompletedResultFileBlocker): string {
    const fileList = blocker.resultPaths.join(', ')

    return [
        `WARNING: Existing ${blocker.mode} state/result file still exists: ${fileList}.`,
        'Task outputs are isolated under output/<task_code>; create a new task or resume a matching task from the task selector.',
    ].join(' ')
}
