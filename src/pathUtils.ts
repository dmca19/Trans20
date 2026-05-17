import { realpath, stat } from 'node:fs/promises'
import path from 'node:path'

import type { Stats } from 'node:fs'

export type ProjectFile = {
    resolvedPath: string
    realFilePath: string
    relativePath: string
    normalizedRelativePath: string
    fileStat: Stats
}

export type ResolveProjectFileOptions = {
    rootPathMessage?: string
    realPathMessage?: string
}

export type ProjectPath = {
    resolvedPath: string
    relativePath: string
    normalizedRelativePath: string
}

export type ResolveProjectPathOptions = {
    rootPathMessage?: string
    rejectRootPathMessage?: string
}

export function resolveProjectPath (
    root: string,
    requestedPath: string,
    options: ResolveProjectPathOptions = {},
): ProjectPath {
    const resolvedPath = path.resolve(root, requestedPath)
    const relativePath = path.relative(root, resolvedPath)

    if (isPathOutsideRoot(relativePath)) {
        throw new Error(options.rootPathMessage ?? `Path is outside the project root: ${requestedPath}`)
    }

    if (relativePath === '') {
        throw new Error(options.rejectRootPathMessage ?? `Path is outside the project root: ${requestedPath}`)
    }

    const normalizedRelativePath = toPosixPath(relativePath)

    return {
        resolvedPath,
        relativePath: normalizedRelativePath,
        normalizedRelativePath,
    }
}

export async function resolveProjectFile (
    root: string,
    requestedPath: string,
    options: ResolveProjectFileOptions = {},
): Promise<ProjectFile> {
    const projectPath = resolveProjectPath(root, requestedPath, {
        rootPathMessage: options.rootPathMessage,
        rejectRootPathMessage: options.rootPathMessage,
    })

    const realRoot = await realpath(root)
    const realFilePath = await realpath(projectPath.resolvedPath)
    const realRelativePath = path.relative(realRoot, realFilePath)

    if (isPathOutsideRoot(realRelativePath) || realRelativePath === '') {
        throw new Error(options.realPathMessage ?? `Path resolves outside the project root: ${requestedPath}`)
    }

    const fileStat = await stat(realFilePath)

    if (!fileStat.isFile()) {
        throw new Error(`${requestedPath} is not a file.`)
    }

    const normalizedRelativePath = toPosixPath(realRelativePath)

    return {
        resolvedPath: projectPath.resolvedPath,
        realFilePath,
        relativePath: normalizedRelativePath,
        normalizedRelativePath,
        fileStat,
    }
}

export function normalizePathId (value: string): string {
    return value.replace(/\\/g, '/').replace(/^\.\//, '')
}

export function normalizeProjectFileId (value: string): string {
    return normalizePathId(toPosixPath(value))
}

export function toPosixPath (filePath: string): string {
    return filePath.split(path.sep).join('/')
}

function isPathOutsideRoot (relativePath: string): boolean {
    return relativePath === '..' || relativePath.startsWith(`..${path.sep}`) || path.isAbsolute(relativePath)
}
