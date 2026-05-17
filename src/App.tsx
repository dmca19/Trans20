import { execFile } from 'node:child_process'
import { realpath, stat } from 'node:fs/promises'
import path from 'node:path'

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Box, Text, useApp, useInput, useStdin, useStdout, useWindowSize } from 'ink'
import { ScrollBar } from '@byteland/ink-scroll-bar'
import cliTruncate from 'cli-truncate'
import { ControlledScrollView } from 'ink-scroll-view'
import stringWidth from 'string-width'
import wrapAnsi from 'wrap-ansi'
// Currently dead code: Ask mode is disabled; kept for possible future reuse.
import type { BaseMessageLike } from '@langchain/core/messages'

import {
    type AgentApiEvent,
    type AgentTokenUsageRecord,
    // Currently dead code: Ask mode is disabled; kept for possible future reuse.
    createProjectAgent,
    exportTranslationResult,
    type GlossaryReviewPaneUpdate,
    type GlossaryReviewStatus,
    type GlossaryWorkerPaneUpdate,
    type GlossaryWorkerStatus,
    // Currently dead code: Ask mode is disabled; kept for possible future reuse.
    getLastMessageContent,
    createTaskRunContext,
    // Currently dead code: Ask mode is disabled; kept for possible future reuse.
    invokeProjectAgent,
    type ProjectAgent,
    runGlossaryPreflight,
    runGlossaryWorkers,
    runTranslationPreflight,
    runTranslationWorkers,
    testConfiguredModel,
} from './agent.js'
import {
    applyRecommendedAgentConfig,
    createDefaultAgentConfig,
    type AgentConfig,
    getRecommendedAgentConfig,
    loadAgentConfig,
    loadEditableAgentConfig,
    saveAgentConfig,
} from './config.js'
import { flushAllGlossaryStores, flushGlossaryState, type GlossaryPlan } from './glossaryStore.js'
import { loadPreferences, savePreferences, type Preferences } from './preferences.js'
import { createRunLogger, type RunLogger } from './runLogger.js'
import {
    formatSettingsDisplayValue,
    formatSettingsRawValue,
    getSettingsEditOptions,
    getSettingsFieldDescription,
    getSettingsFieldLabel,
    getSettingsFieldRawValue,
    maxSettingsFieldCount,
    parseSettingsEditValue,
    settingsCategories,
    settingsFieldDefinitions,
    type SettingsFieldId,
} from './settingsDefinitions.js'
import {
    applyTaskConfigSnapshot,
    compareTaskConfigSnapshot,
    createGlossarySourceFromTask,
    createTask,
    createTaskRunPaths,
    isCompleteTaskConfigSnapshot,
    listGlossarySourceTasks,
    listTasks,
    loadTask,
    markTaskInterrupted,
    updateTask,
    validateGlossarySource,
    type ProjectTask,
    type TaskConfigDifference,
    type TaskGlossarySource,
    type TaskStage,
    type TaskStatus,
} from './taskStore.js'
import { flushAllTranslationStores, type TranslationExportResult, type TranslationPreflight } from './translationStore.js'
import { DEFAULT_TUI_LANGUAGE, type TuiLanguage } from './tuiLanguage.js'
import { getTuiText, type TuiText } from './tuiText.js'

import type { ToolCallEvent } from './inspectionTools.js'

export type AppMode = 'glossary'|'translation'

// Differs from TaskStage: AppStage is UI-only.
// Ask mode stages are currently dead code; kept for possible future reuse.
type AppStage =
    'checking-setup'
    |'start-menu'
    |'setup-wizard'
    |'settings'
    |'settings-edit'
    |'select-directory'
    |'select-task'
    |'select-glossary'
    |'confirm-config'
    |'initializing'
    |'preflighting'
    |'extracting'
    |'exporting'
    |'task-complete'
    |'ready'
    |'running'

export type TranscriptEntry = {
    id: string
    kind: 'user'|'assistant'|'tool'|'system'|'error'
    text: string
}

export type DisplayLine = {
    id: string
    text: string
    color?: string
    bold?: boolean
    dimColor?: boolean
}

type PaneLine = DisplayLine

export type MouseWheelEvent = {
    direction: 'up'|'down'
    x: number|null
    y: number|null
}

export type MouseInputEvent = {
    kind: 'press'|'release'|'move'|'wheel'
    button: 'left'|'other'|null
    direction?: 'up'|'down'
    x: number
    y: number
}

type MousePositionEvent = {
    x: number
    y: number
}

export type DualPaneLayout = {
    contentStartRow: number
    contentEndRow: number
    leftStartColumn: number
    leftEndColumn: number
    rightStartColumn: number
    rightEndColumn: number
    paneWidth: number
    transcriptWidth: number
}

type WorkerPaneGridLayout = {
    contentRows: number
    contentWidth: number
    hasScrollBar: boolean
    paneWidth: number
    panesPerRow: number
}

export type PaneScrollTarget =
    | { kind: 'worker-pane', slotIndex: number }
    | { kind: 'review-pane' }
    | { kind: 'left-pane' }
    | { kind: 'transcript' }

export type TaskSelectionDialogLayout = {
    startColumn: number
    endColumn: number
    startRow: number
    endRow: number
    optionStartRow: number
    optionEndRow: number
    buttonRow: number
    confirmButtonStartColumn: number
    confirmButtonEndColumn: number
    cancelButtonStartColumn: number
    cancelButtonEndColumn: number
    changeDirectoryButtonStartColumn: number
    changeDirectoryButtonEndColumn: number
    choiceCount: number
}

export type TaskSelectionMouseTarget =
    | { kind: 'choice', index: number }
    | { kind: 'confirm' }
    | { kind: 'cancel' }
    | { kind: 'change-directory' }

export type ConfigMismatchChoice = 'use-task-config'|'back'

export type ConfigMismatchLayout = {
    startColumn: number
    endColumn: number
    startRow: number
    endRow: number
    buttonRow: number
    useTaskConfigButtonStartColumn: number
    useTaskConfigButtonEndColumn: number
    backButtonStartColumn: number
    backButtonEndColumn: number
}

export type ConfigMismatchMouseTarget = { kind: 'button', choice: ConfigMismatchChoice }

export type TaskCompleteLayout = {
    startColumn: number
    endColumn: number
    startRow: number
    endRow: number
    buttonRow: number
    buttonStartColumn: number
    buttonEndColumn: number
}

export type TaskCompleteMouseTarget = { kind: 'button' }

type StartMenuChoice = 'glossary'|'translation'|'settings'|'setupWizard'

export type StartMenuLayout = {
    startColumn: number
    endColumn: number
    startRow: number
    endRow: number
    choiceRows: Array<{
        choice: StartMenuChoice
        row: number
    }>
}

export type StartMenuMouseTarget = { kind: 'choice', choice: StartMenuChoice }

type SettingsFocusArea = 'categories'|'fields'|'buttons'
type SettingsButton = 'save'|'back'

type SettingsEditState = {
    field: SettingsFieldId
    input: string
    selectedOptionIndex: number
}

type ClipboardPasteContext = {
    key: string
    version: number
}

export type PendingGlossarySelection = {
    task: ProjectTask
    config: AgentConfig
    resumed: boolean
}

export type SettingsLayout = {
    startColumn: number
    endColumn: number
    startRow: number
    endRow: number
    leftStartColumn: number
    leftEndColumn: number
    rightStartColumn: number
    rightEndColumn: number
    categoryStartRow: number
    categoryEndRow: number
    fieldStartRow: number
    fieldEndRow: number
    fieldHitEndRow: number
    buttonRow: number
    saveButtonStartColumn: number
    saveButtonEndColumn: number
    backButtonStartColumn: number
    backButtonEndColumn: number
}

export type SettingsMouseTarget =
    | { kind: 'category', index: number }
    | { kind: 'field', index: number }
    | { kind: 'button', button: SettingsButton }

export type SettingsEditLayout = {
    startColumn: number
    endColumn: number
    startRow: number
    endRow: number
    optionStartRow: number
    optionEndRow: number
    buttonRow: number
    confirmButtonStartColumn: number
    confirmButtonEndColumn: number
    cancelButtonStartColumn: number
    cancelButtonEndColumn: number
}

export type SettingsEditMouseTarget =
    | { kind: 'option', index: number }
    | { kind: 'confirm' }
    | { kind: 'cancel' }

type SetupWizardStepId = 'welcome'|'api'|'apiTest'|'recommended'|'project'
type SetupWizardFocusArea = 'fields'|'buttons'
type SetupWizardPrimaryButton = 'next'|'finish'|'applyRecommended'
type SetupWizardButton = 'previous'|'retest'|'skipRecommended'|SetupWizardPrimaryButton
type SetupWizardButtonFocus = 'previous'|'retest'|'secondary'|'primary'
type SettingsEditReturnStage = 'settings'|'setup-wizard'
type SetupWizardApiTestState =
    | { status: 'idle' }
    | { status: 'running' }
    | { status: 'succeeded', response: string }
    | { status: 'failed', error: string }

export type SetupWizardLayout = {
    startColumn: number
    endColumn: number
    startRow: number
    endRow: number
    leftStartColumn: number
    leftEndColumn: number
    rightStartColumn: number
    rightEndColumn: number
    stepStartRow: number
    stepEndRow: number
    fieldStartRow: number
    fieldEndRow: number
    buttonRow: number
    previousButtonStartColumn: number
    previousButtonEndColumn: number
    retestButtonStartColumn: number|null
    retestButtonEndColumn: number|null
    secondaryButtonStartColumn: number|null
    secondaryButtonEndColumn: number|null
    primaryButtonStartColumn: number
    primaryButtonEndColumn: number
    primaryButton: SetupWizardPrimaryButton
    previousEnabled: boolean
}

export type SetupWizardMouseTarget =
    | { kind: 'field', index: number }
    | { kind: 'button', button: SetupWizardButton }

type TaskSelectionButtonTarget = Exclude<TaskSelectionMouseTarget, { kind: 'choice' }>

type DialogButtonBounds = {
    startColumn: number
    endColumn: number
}

type ButtonRowBounds = DialogButtonBounds[]

type TokenUsageState = {
    inputTokens: number
    outputTokens: number
    totalTokens: number
}

type TokenPulseState = {
    inputDelta: number
    outputDelta: number
    totalDelta: number
}

type ApiStatusState = {
    label: string
    status: AgentApiEvent['status']
    attempt: number
    maxAttempts: number
}

type SetupWizardStepDefinition = {
    id: SetupWizardStepId
    fields: SettingsFieldId[]
}

const SETUP_WIZARD_STEPS: SetupWizardStepDefinition[] = [
    {
        id: 'welcome',
        fields: [],
    },
    {
        id: 'api',
        fields: ['url', 'key', 'model', 'reasoningEffort'],
    },
    {
        id: 'apiTest',
        fields: [],
    },
    {
        id: 'recommended',
        fields: [],
    },
    {
        id: 'project',
        fields: ['tuiLanguage', 'manualTransFile'],
    },
]

const defaultDirectory = process.cwd()

const DEFAULT_WORKER_PANE_COUNT = 4
const MAX_TRANSCRIPT_ENTRIES = 500
const SCROLL_STEP_LINES = 3
const LAYOUT_GAP_COLUMNS = 0
const ROOT_PADDING_COLUMNS = 1
const HEADER_ROWS = 2
const WORKER_PANE_MIN_WIDTH = 38
const WORKER_PANE_HEIGHT = 11
const WORKER_PANE_MARGIN_RIGHT = 1
const WORKER_PANE_MARGIN_BOTTOM = 1
const WORKER_GRID_TITLE_ROWS = 1
const MAX_VISIBLE_TASK_CHOICES = 5
const TASK_DIALOG_SIDE_MARGIN_COLUMNS = 1
const TASK_DIALOG_MAX_WIDTH = 96
const TASK_DIALOG_MIN_WIDE_WIDTH = 44
const TASK_DIALOG_BORDER_ROWS = 2
const TASK_DIALOG_PADDING_Y = 1
const TASK_DIALOG_VERTICAL_PADDING_ROWS = TASK_DIALOG_PADDING_Y * 2
const TASK_DIALOG_HEADER_ROWS = 3
const TASK_DIALOG_OPTION_MARGIN_TOP_ROWS = 1
const TASK_DIALOG_BUTTON_MARGIN_TOP_ROWS = 1
const TASK_DIALOG_BUTTON_ROWS = 1
const TASK_DIALOG_HELP_ROWS = 1
const TASK_DIALOG_FOOTER_ROWS = TASK_DIALOG_BUTTON_MARGIN_TOP_ROWS + TASK_DIALOG_BUTTON_ROWS + TASK_DIALOG_HELP_ROWS
const TASK_DIALOG_BUTTON_GAP_COLUMNS = 1
const START_MENU_CHOICES: StartMenuChoice[] = [
    'glossary',
    'translation',
    'settings',
    'setupWizard',
]
const START_MENU_DIALOG_WIDTH = 36
const START_MENU_BORDER_ROWS = 2
const START_MENU_PADDING_Y = 1
const START_MENU_VERTICAL_PADDING_ROWS = START_MENU_PADDING_Y * 2
const START_MENU_HEADER_ROWS = 2
const START_MENU_OPTION_MARGIN_TOP_ROWS = 1
const START_MENU_OPTION_GAP_ROWS = 1
const SETTINGS_DIALOG_SIDE_MARGIN_COLUMNS = 1
const SETTINGS_DIALOG_MAX_WIDTH = 118
const SETTINGS_DIALOG_MIN_WIDTH = 48
const SETTINGS_DIALOG_MIN_ROWS = 10
const SETTINGS_DIALOG_BORDER_ROWS = 2
const SETTINGS_DIALOG_PADDING_Y = 1
const SETTINGS_DIALOG_VERTICAL_PADDING_ROWS = SETTINGS_DIALOG_PADDING_Y * 2
const SETTINGS_DIALOG_HEADER_ROWS = 3
const SETTINGS_DIALOG_LIST_MARGIN_TOP_ROWS = 1
const SETTINGS_DIALOG_BUTTON_MARGIN_TOP_ROWS = 1
const SETTINGS_DIALOG_BUTTON_ROWS = 1
const SETTINGS_DIALOG_FOOTER_ROWS = 1
const SETTINGS_DIALOG_CATEGORY_WIDTH = 18
const SETTINGS_EDIT_DIALOG_MAX_WIDTH = 76
const SETTINGS_EDIT_DIALOG_MIN_WIDTH = 42
const SETTINGS_EDIT_MAX_VISIBLE_OPTIONS = 6
const SETTINGS_EDIT_DIALOG_HEADER_ROWS = 4
const SETTINGS_EDIT_DIALOG_INPUT_MARGIN_TOP_ROWS = 1
const SETTINGS_EDIT_DIALOG_BUTTON_MARGIN_TOP_ROWS = 1
const SETTINGS_EDIT_DIALOG_BUTTON_ROWS = 1
const SETUP_WIZARD_DIALOG_SIDE_MARGIN_COLUMNS = 1
const SETUP_WIZARD_DIALOG_MAX_WIDTH = 106
const SETUP_WIZARD_DIALOG_MIN_WIDTH = 52
const SETUP_WIZARD_DIALOG_MIN_ROWS = 12
const SETUP_WIZARD_BORDER_ROWS = 2
const SETUP_WIZARD_PADDING_Y = 1
const SETUP_WIZARD_VERTICAL_PADDING_ROWS = SETUP_WIZARD_PADDING_Y * 2
const SETUP_WIZARD_HEADER_ROWS = 3
const SETUP_WIZARD_LIST_MARGIN_TOP_ROWS = 1
const SETUP_WIZARD_BUTTON_MARGIN_TOP_ROWS = 1
const SETUP_WIZARD_BUTTON_ROWS = 1
const SETUP_WIZARD_FOOTER_ROWS = 1
const SETUP_WIZARD_STEP_WIDTH = 20
const SETUP_WIZARD_STEP_HEADER_ROWS = 3
const CONFIG_MISMATCH_MAX_VISIBLE_DIFFERENCES = 6
const CONFIG_MISMATCH_BORDER_ROWS = 2
const CONFIG_MISMATCH_PADDING_Y = 1
const CONFIG_MISMATCH_VERTICAL_PADDING_ROWS = CONFIG_MISMATCH_PADDING_Y * 2
const CONFIG_MISMATCH_HEADER_ROWS = 2
const CONFIG_MISMATCH_DIFFERENCE_MARGIN_TOP_ROWS = 1
const CONFIG_MISMATCH_MIN_DIFFERENCE_ROWS = 1
const CONFIG_MISMATCH_BUTTON_MARGIN_TOP_ROWS = 1
const CONFIG_MISMATCH_BUTTON_ROWS = 1
const CONFIG_MISMATCH_FOOTER_ROWS = 1
const CONFIG_MISMATCH_VISIBLE_DIFFERENCE_RESERVED_ROWS = CONFIG_MISMATCH_BORDER_ROWS
    + CONFIG_MISMATCH_VERTICAL_PADDING_ROWS
    + CONFIG_MISMATCH_HEADER_ROWS
    + CONFIG_MISMATCH_DIFFERENCE_MARGIN_TOP_ROWS
    + CONFIG_MISMATCH_BUTTON_MARGIN_TOP_ROWS
const TASK_COMPLETE_BORDER_ROWS = 2
const TASK_COMPLETE_PADDING_Y = 1
const TASK_COMPLETE_VERTICAL_PADDING_ROWS = TASK_COMPLETE_PADDING_Y * 2
const TASK_COMPLETE_HEADER_ROWS = 3
const TASK_COMPLETE_BUTTON_MARGIN_TOP_ROWS = 1
const TASK_COMPLETE_BUTTON_ROWS = 1
const TASK_COMPLETE_FOOTER_ROWS = 1
export const CTRL_C_EXIT_WINDOW_MS = 2000
export const CTRL_C_HARD_EXIT_CODE = 130
const HARD_EXIT_TUI_TEARDOWN_DELAY_MS = 25
const TOKEN_ANIMATION_MS = 700
const TERMINAL_MOUSE_ENABLE_SEQUENCE = '\u001B[?1007l\u001B[?1006h\u001B[?1000h\u001B[?1002h\u001B[?1003h'
const TERMINAL_MOUSE_DISABLE_SEQUENCE = '\u001B[?1003l\u001B[?1002l\u001B[?1000l\u001B[?1006l\u001B[?1007h'
const TERMINAL_ALTERNATE_SCREEN_DISABLE_SEQUENCE = '\u001B[?1049l'
const MOUSE_MODE_ENABLE_DELAYS_MS = [0, 25, 100, 500]
const READ_CLIPBOARD_TIMEOUT_MS = 1500
const READ_CLIPBOARD_MAX_BUFFER = 2 * 1024 * 1024
const CLIPBOARD_MESSAGE_VISIBLE_MS = 2500
const emptyTokenUsage: TokenUsageState = {
    inputTokens: 0,
    outputTokens: 0,
    totalTokens: 0,
}
const emptyTokenPulse: TokenPulseState = {
    inputDelta: 0,
    outputDelta: 0,
    totalDelta: 0,
}

const initialWorkerPanes = (count = DEFAULT_WORKER_PANE_COUNT): GlossaryWorkerPaneUpdate[] => Array.from({ length: count }, (_item, index) => ({
    slotIndex: index,
    lifecycle: 0,
    status: 'idle',
    batchId: null,
    batchNumber: null,
    totalBatches: 0,
    batchStartIndex: null,
    batchEndIndex: null,
    keyCount: 0,
}))

type ReviewPaneState = GlossaryReviewPaneUpdate & {
    events: string[]
}

const initialReviewPane = (): ReviewPaneState => ({
    status: 'idle',
    reviewWindowId: null,
    batchNumbers: [],
    pendingBatchCount: 0,
    events: [],
})

function createEmptyPreferences (): Preferences {
    return {
        setupWizardCompleted: false,
        tuiLanguage: DEFAULT_TUI_LANGUAGE,
    }
}

export type AppProps = {
    initialMode?: AppMode
    language?: TuiLanguage
}

export function App ({
    initialMode = 'glossary',
    language = 'en',
}: AppProps = {}): React.ReactElement {
    const { exit } = useApp()
    const [tuiLanguage, setTuiLanguage] = useState<TuiLanguage>(language)
    const text = useMemo(() => getTuiText(tuiLanguage), [tuiLanguage])
    const { rows, columns } = useWindowSize()
    const { stdin } = useStdin()
    const { stdout } = useStdout()
    const loggerRef = useRef<RunLogger|null>(null)
    const exitingRef = useRef(false)
    const exitPromptTimerRef = useRef<NodeJS.Timeout|null>(null)
    const clipboardMessageTimerRef = useRef<NodeJS.Timeout|null>(null)
    const clipboardPasteContextRef = useRef<ClipboardPasteContext>({
        key: getClipboardPasteContextKey('checking-setup', null),
        version: 0,
    })
    const seenTokenUsageIdsRef = useRef(new Set<string>())
    const recentMouseWheelEventsRef = useRef(new Map<string, number>())
    const lastRawMouseWheelAtRef = useRef(0)
    const lastCtrlCAtRef = useRef<number|null>(null)
    const tokenPulseTimerRef = useRef<NodeJS.Timeout|null>(null)
    const activeTaskRef = useRef<ProjectTask|null>(null)
    const effectiveConfigRef = useRef<AgentConfig|null>(null)
    const forcedResumeConfigRef = useRef<{ taskCode: string, config: AgentConfig }|null>(null)
    const pendingGlossarySelectionRef = useRef<PendingGlossarySelection|null>(null)
    const runSelectedTaskRef = useRef<((task: ProjectTask, config: AgentConfig, resumed: boolean, glossarySource: TaskGlossarySource|null) => Promise<void>)|null>(null)
    const startupSetupCheckStartedRef = useRef(false)
    const setupWizardApiTestRunRef = useRef(0)
    const [stage, setStage] = useState<AppStage>('checking-setup')
    const [input, setInput] = useState('')
    const [exitPromptVisible, setExitPromptVisible] = useState(false)
    const [runDirectory, setRunDirectory] = useState(defaultDirectory)
    const [mode, setMode] = useState<AppMode>(initialMode)
    const [selectedStartMenuIndex, setSelectedStartMenuIndex] = useState(0)
    const [hoveredStartMenuChoice, setHoveredStartMenuChoice] = useState<StartMenuChoice|null>(null)
    const [taskChoices, setTaskChoices] = useState<ProjectTask[]>([])
    const [glossaryChoices, setGlossaryChoices] = useState<ProjectTask[]>([])
    const [selectedTaskIndex, setSelectedTaskIndex] = useState(0)
    const [selectedGlossaryIndex, setSelectedGlossaryIndex] = useState(0)
    const [hoveredTaskSelectionTarget, setHoveredTaskSelectionTarget] = useState<TaskSelectionMouseTarget|null>(null)
    const [taskSelectionKeyboardTarget, setTaskSelectionKeyboardTarget] = useState<TaskSelectionMouseTarget|null>(null)
    const [settingsConfig, setSettingsConfig] = useState<AgentConfig>(() => createDefaultAgentConfig())
    const [settingsSavedConfig, setSettingsSavedConfig] = useState<AgentConfig>(() => createDefaultAgentConfig())
    const [settingsPreferences, setSettingsPreferences] = useState<Preferences>(() => createEmptyPreferences())
    const [settingsSavedPreferences, setSettingsSavedPreferences] = useState<Preferences>(() => createEmptyPreferences())
    const [settingsLoadError, setSettingsLoadError] = useState<string|null>(null)
    const [settingsSaveMessage, setSettingsSaveMessage] = useState<string|null>(null)
    const [selectedSettingsCategoryIndex, setSelectedSettingsCategoryIndex] = useState(0)
    const [selectedSettingsFieldIndex, setSelectedSettingsFieldIndex] = useState(0)
    const [settingsFocusArea, setSettingsFocusArea] = useState<SettingsFocusArea>('categories')
    const [selectedSettingsButtonIndex, setSelectedSettingsButtonIndex] = useState(0)
    const [hoveredSettingsTarget, setHoveredSettingsTarget] = useState<SettingsMouseTarget|null>(null)
    const [settingsEdit, setSettingsEdit] = useState<SettingsEditState|null>(null)
    const [settingsEditError, setSettingsEditError] = useState<string|null>(null)
    const [hoveredSettingsEditTarget, setHoveredSettingsEditTarget] = useState<SettingsEditMouseTarget|null>(null)
    const [settingsEditKeyboardTarget, setSettingsEditKeyboardTarget] = useState<SettingsEditMouseTarget|null>(null)
    const [settingsEditReturnStage, setSettingsEditReturnStage] = useState<SettingsEditReturnStage>('settings')
    const [clipboardMessage, setClipboardMessage] = useState<string|null>(null)
    const [selectedSetupWizardStepIndex, setSelectedSetupWizardStepIndex] = useState(0)
    const [selectedSetupWizardFieldIndex, setSelectedSetupWizardFieldIndex] = useState(0)
    const [setupWizardFocusArea, setSetupWizardFocusArea] = useState<SetupWizardFocusArea>('fields')
    const [setupWizardButtonFocus, setSetupWizardButtonFocus] = useState<SetupWizardButtonFocus>('primary')
    const [hoveredSetupWizardTarget, setHoveredSetupWizardTarget] = useState<SetupWizardMouseTarget|null>(null)
    const [setupWizardFirstRun, setSetupWizardFirstRun] = useState(false)
    const [setupWizardApiTest, setSetupWizardApiTest] = useState<SetupWizardApiTestState>({ status: 'idle' })
    const [activeTask, setActiveTask] = useState<ProjectTask|null>(null)
    const [pendingGlossarySelection, setPendingGlossarySelection] = useState<PendingGlossarySelection|null>(null)
    const [pendingConfigResume, setPendingConfigResume] = useState<{
        task: ProjectTask
        currentConfig: AgentConfig
        taskConfig: AgentConfig
        differences: TaskConfigDifference[]
    }|null>(null)
    const [hoveredConfigMismatchChoice, setHoveredConfigMismatchChoice] = useState<ConfigMismatchChoice|null>(null)
    const [configMismatchKeyboardChoice, setConfigMismatchKeyboardChoice] = useState<ConfigMismatchChoice|null>(null)
    const [hoveredTaskCompleteTarget, setHoveredTaskCompleteTarget] = useState<TaskCompleteMouseTarget|null>(null)
    // Currently dead code: Ask mode is disabled; kept for possible future reuse.
    const [agent, setAgent] = useState<ProjectAgent|null>(null)
    // Currently dead code: Ask mode is disabled; kept for possible future reuse.
    const [messages, setMessages] = useState<BaseMessageLike[]>([])
    const [workerPanes, setWorkerPanes] = useState<GlossaryWorkerPaneUpdate[]>(initialWorkerPanes)
    const [reviewPane, setReviewPane] = useState<ReviewPaneState>(initialReviewPane)
    const [transcript, setTranscript] = useState<TranscriptEntry[]>([
        {
            id: 'welcome',
            kind: 'system',
            text: text.message.loadingRunDirectory(defaultDirectory),
        },
    ])
    const [scrollOffsetFromBottom, setScrollOffsetFromBottom] = useState(0)
    const [paneScrollOffsetFromTop, setPaneScrollOffsetFromTop] = useState(0)
    const [workerPaneScrollOffsets, setWorkerPaneScrollOffsets] = useState<Record<number, number>>({})
    const [reviewPaneScrollOffset, setReviewPaneScrollOffset] = useState(0)
    const [tokenUsage, setTokenUsage] = useState<TokenUsageState>(emptyTokenUsage)
    const [tokenPulse, setTokenPulse] = useState<TokenPulseState>(emptyTokenPulse)

    useEffect(() => {
        activeTaskRef.current = activeTask
    }, [activeTask])
    useEffect(() => {
        pendingGlossarySelectionRef.current = pendingGlossarySelection
    }, [pendingGlossarySelection])
    const [apiStatus, setApiStatus] = useState<ApiStatusState|null>(null)
    const clipboardPasteContextKey = getClipboardPasteContextKey(stage, settingsEdit)

    useEffect(() => {
        const currentContext = clipboardPasteContextRef.current

        if (currentContext.key !== clipboardPasteContextKey) {
            clipboardPasteContextRef.current = {
                key: clipboardPasteContextKey,
                version: currentContext.version + 1,
            }
        }
    }, [clipboardPasteContextKey])

    useEffect(() => {
        if (typeof performance.clearMeasures !== 'function' || typeof performance.clearMarks !== 'function') {
            return
        }

        const clearReactRenderMeasurements = (): void => {
            performance.clearMeasures()
            performance.clearMarks()
        }

        clearReactRenderMeasurements()
        const timer = setInterval(clearReactRenderMeasurements, 5000)

        return () => {
            clearInterval(timer)
            clearReactRenderMeasurements()
        }
    }, [])

    const appendTranscript = useCallback((entry: Omit<TranscriptEntry, 'id'>) => {
        const nextEntry = {
            ...entry,
            id: `${Date.now()}-${Math.random()}`,
        }

        loggerRef.current?.log('transcript', nextEntry)
        setTranscript(current => [
            ...current,
            nextEntry,
        ].slice(-MAX_TRANSCRIPT_ENTRIES))
    }, [])

    const handleToolEvent = useCallback((event: ToolCallEvent) => {
        loggerRef.current?.log('tool_event', event)

        const statusPrefix = text.states.toolStatus[event.status]
        const details = event.status === 'started'
            ? event.input
            : event.status === 'completed'
                ? event.output ?? ''
                : event.error ?? ''

        appendTranscript({
            kind: 'tool',
            text: `[${statusPrefix}] ${event.toolName}\n${truncateForDisplay(details, loggerRef.current?.filePath != null, text)}`,
        })
    }, [appendTranscript, text])

    const handleWorkerUpdate = useCallback((update: GlossaryWorkerPaneUpdate) => {
        loggerRef.current?.log('worker_update', update)
        setWorkerPanes(current => {
            const existingPane = current.find(pane => pane.slotIndex === update.slotIndex)

            if (!existingPane) {
                return [...current, update].sort((left, right) => left.slotIndex - right.slotIndex)
            }

            return current.map(pane => (pane.slotIndex === update.slotIndex ? { ...pane, ...update } : pane))
        })
    }, [])

    const handleReviewUpdate = useCallback((update: GlossaryReviewPaneUpdate) => {
        loggerRef.current?.log('review_update', update)
        setReviewPane(current => {
            const eventText = update.lastTool
                ? [
                    update.lastTool,
                    update.toolInput ? `input: ${truncatePaneText(update.toolInput)}` : '',
                    update.toolOutput ? `result: ${truncatePaneText(update.toolOutput)}` : '',
                ].filter(Boolean).join('\n')
                : null

            return {
                ...current,
                ...update,
                events: eventText ? [...current.events, eventText].slice(-8) : current.events,
            }
        })
    }, [])

    const applyTokenUsageRecords = useCallback((records: AgentTokenUsageRecord[]|undefined) => {
        if (!records || records.length === 0) {
            return
        }

        let inputDelta = 0
        let outputDelta = 0
        let totalDelta = 0

        for (const record of records) {
            if (seenTokenUsageIdsRef.current.has(record.id)) {
                continue
            }

            seenTokenUsageIdsRef.current.add(record.id)
            inputDelta += record.inputTokens
            outputDelta += record.outputTokens
            totalDelta += record.totalTokens
        }

        if (inputDelta === 0 && outputDelta === 0 && totalDelta === 0) {
            return
        }

        setTokenUsage(current => ({
            inputTokens: current.inputTokens + inputDelta,
            outputTokens: current.outputTokens + outputDelta,
            totalTokens: current.totalTokens + totalDelta,
        }))
        setTokenPulse({
            inputDelta,
            outputDelta,
            totalDelta,
        })

        if (tokenPulseTimerRef.current) {
            clearTimeout(tokenPulseTimerRef.current)
        }

        tokenPulseTimerRef.current = setTimeout(() => {
            setTokenPulse(emptyTokenPulse)
            tokenPulseTimerRef.current = null
        }, TOKEN_ANIMATION_MS)
    }, [])

    const showClipboardMessage = useCallback((message: string) => {
        setClipboardMessage(message)

        if (clipboardMessageTimerRef.current) {
            clearTimeout(clipboardMessageTimerRef.current)
        }

        clipboardMessageTimerRef.current = setTimeout(() => {
            setClipboardMessage(null)
            clipboardMessageTimerRef.current = null
        }, CLIPBOARD_MESSAGE_VISIBLE_MS)
    }, [])

    const handleApiEvent = useCallback((event: AgentApiEvent) => {
        loggerRef.current?.debug('api_event', event)
        setApiStatus({
            label: event.label,
            status: event.status,
            attempt: event.attempt,
            maxAttempts: event.maxAttempts,
        })
        applyTokenUsageRecords(event.usageRecords)

        if (event.label.startsWith('review-agent')) {
            setReviewPane(current => ({
                ...current,
                events: [...current.events, formatApiEventForPane(event, text)].slice(-8),
            }))
        }

        if (event.status === 'retrying') {
            appendTranscript({
                kind: 'system',
                text: text.message.apiRetry(event.label, event.attempt, event.maxAttempts, event.error?.message ?? ''),
            })
        }
    }, [appendTranscript, applyTokenUsageRecords, text])

    const flushAndExit = useCallback((inputName: string) => {
        if (exitingRef.current) {
            return
        }

        exitingRef.current = true
        loggerRef.current?.log('exit_requested', { input: inputName })

        void (async () => {
            try {
                await flushAllGlossaryStores()
                await flushAllTranslationStores()
                const pendingGlossarySelection = pendingGlossarySelectionRef.current
                if (pendingGlossarySelection && pendingGlossarySelection.task.status !== 'completed') {
                    await updateTask(runDirectory, pendingGlossarySelection.task.task_code, { status: 'interrupted' })
                    pendingGlossarySelectionRef.current = null
                }

                const task = activeTaskRef.current
                if (task) {
                    const currentTask = await loadTask(runDirectory, task.task_code).catch(() => task)

                    if (currentTask.status !== 'completed') {
                        await updateTask(runDirectory, task.task_code, { status: 'interrupted' })
                    }
                }
                loggerRef.current?.log('glossary_flush_completed_on_exit', {})
            } catch (error) {
                loggerRef.current?.log('state_flush_failed_on_exit', {
                    error: error instanceof Error ? error.message : String(error),
                })
            } finally {
                loggerRef.current?.close()
                exit()
            }
        })()
    }, [exit, runDirectory])

    const requestExit = useCallback(() => {
        const now = Date.now()
        const nextState = getCtrlCExitState(lastCtrlCAtRef.current, now)
        lastCtrlCAtRef.current = nextState.lastCtrlCAt

        if (nextState.shouldExit) {
            if (exitPromptTimerRef.current) {
                clearTimeout(exitPromptTimerRef.current)
                exitPromptTimerRef.current = null
            }

            if (clipboardMessageTimerRef.current) {
                clearTimeout(clipboardMessageTimerRef.current)
                clipboardMessageTimerRef.current = null
            }

            if (tokenPulseTimerRef.current) {
                clearTimeout(tokenPulseTimerRef.current)
                tokenPulseTimerRef.current = null
            }

            hardExitNow({
                inputName: 'ctrl+c',
                logger: loggerRef.current,
                restoreTerminal: () => {
                    stdout.write(`${TERMINAL_MOUSE_DISABLE_SEQUENCE}${TERMINAL_ALTERNATE_SCREEN_DISABLE_SEQUENCE}`)
                },
                exit,
            })
            return
        }

        setExitPromptVisible(true)
        if (exitPromptTimerRef.current) {
            clearTimeout(exitPromptTimerRef.current)
        }
        exitPromptTimerRef.current = setTimeout(() => {
            setExitPromptVisible(false)
            exitPromptTimerRef.current = null
        }, CTRL_C_EXIT_WINDOW_MS)

        appendTranscript({
            kind: 'system',
            text: text.message.ctrlCAgain,
        })
    }, [appendTranscript, exit, stdout, text])

    const selectDirectory = useCallback(async (directoryInput: string, selectedMode = mode) => {
        setStage('initializing')
        setInput('')
        setWorkerPanes(initialWorkerPanes())
        setReviewPane(initialReviewPane())
        setScrollOffsetFromBottom(0)
        setPaneScrollOffsetFromTop(0)
        setWorkerPaneScrollOffsets({})
        setReviewPaneScrollOffset(0)
        setTokenUsage(emptyTokenUsage)
        setTokenPulse(emptyTokenPulse)
        setApiStatus(null)
        seenTokenUsageIdsRef.current.clear()

        try {
            const directory = await resolveRunDirectory(directoryInput, text)
            const config = await loadAgentConfig()

            if (runDirectory !== directory) {
                await flushAllGlossaryStores()
                await flushAllTranslationStores()
            }

            setWorkerPanes(initialWorkerPanes(selectedMode === 'translation' ? config.translationWorkerParallelism : config.glossaryWorkerParallelism))
            setRunDirectory(directory)
            setActiveTask(null)
            setPendingGlossarySelection(null)
            effectiveConfigRef.current = null
            setPendingConfigResume(null)
            setHoveredConfigMismatchChoice(null)
            setConfigMismatchKeyboardChoice(null)
            setAgent(null)
            setMessages([])
            const tasks = await listTasks(directory, {
                mode: selectedMode,
                resumableOnly: true,
            })
            const initialTranscript: TranscriptEntry[] = [
                {
                    id: 'ready',
                    kind: 'system',
                    text: text.message.runDirectory(directory),
                },
            ]
            setTranscript(initialTranscript)
            setTaskChoices(tasks)
            setGlossaryChoices([])
            setSelectedTaskIndex(0)
            setSelectedGlossaryIndex(0)
            setHoveredTaskSelectionTarget(null)
            setTaskSelectionKeyboardTarget(null)
            setStage('select-task')
        } catch (error) {
            appendTranscript({
                kind: 'error',
                text: error instanceof Error ? error.message : String(error),
            })
            setStage('select-directory')
        }
    }, [appendTranscript, mode, runDirectory, text])

    const initializeAgent = useCallback(async (selectedTaskCode: string|null) => {
        setStage('initializing')
        setInput('')

        let task: ProjectTask|null = null

        try {
            const loadedConfig = await loadAgentConfig()
            let config = loadedConfig

            task = selectedTaskCode
                ? await markTaskInterrupted(runDirectory, selectedTaskCode, loadedConfig.manualTransFile)
                : await createTask(runDirectory, { mode, config })

            if (selectedTaskCode) {
                const forcedResumeConfig = forcedResumeConfigRef.current?.taskCode === task.task_code
                    ? forcedResumeConfigRef.current.config
                    : null

                if (!isCompleteTaskConfigSnapshot(task.config_snapshot)) {
                    appendTranscript({
                        kind: 'error',
                        text: text.message.incompleteLegacyConfig(task.task_code),
                    })
                    flushAndExit('config-snapshot-incomplete')
                    return
                }

                const taskConfig = applyTaskConfigSnapshot(loadedConfig, task.config_snapshot)
                const differences = compareTaskConfigSnapshot(loadedConfig, task.config_snapshot)

                if (differences.length > 0 && !forcedResumeConfig) {
                    setPendingConfigResume({
                        task,
                        currentConfig: loadedConfig,
                        taskConfig,
                        differences,
                    })
                    setHoveredConfigMismatchChoice(null)
                    setConfigMismatchKeyboardChoice(null)
                    setStage('confirm-config')
                    appendTranscript({
                        kind: 'error',
                        text: formatTaskConfigMismatchWarning(task, differences, text),
                    })
                    return
                }

                config = forcedResumeConfig ?? taskConfig
                forcedResumeConfigRef.current = null
            }

            if (mode === 'translation') {
                if (task.glossary_source === undefined) {
                    const glossarySources = await listGlossarySourceTasks(runDirectory, config.manualTransFile)
                    const pendingGlossarySelection = {
                        task,
                        config,
                        resumed: selectedTaskCode !== null,
                    }
                    pendingGlossarySelectionRef.current = pendingGlossarySelection
                    setPendingGlossarySelection(pendingGlossarySelection)
                    setGlossaryChoices(glossarySources)
                    setSelectedGlossaryIndex(0)
                    setHoveredTaskSelectionTarget(null)
                    setTaskSelectionKeyboardTarget(null)
                    setStage('select-glossary')
                    appendTranscript({
                        kind: 'system',
                        text: glossarySources.length === 0
                            ? text.message.noCompletedGlossarySelectNoGlossary
                            : text.message.selectGlossaryForTranslation,
                    })
                    return
                }

                if (task.glossary_source) {
                    await validateGlossarySource(runDirectory, task.glossary_source, config.manualTransFile)
                }
            }

            await runSelectedTaskRef.current?.(task, config, selectedTaskCode !== null, task.glossary_source ?? null)
        } catch (error) {
            loggerRef.current?.log('initialization_failed', {
                error: error instanceof Error ? error.message : String(error),
            })
            if (task) {
                await updateTask(runDirectory, task.task_code, {
                    status: 'failed',
                    last_error: error instanceof Error ? error.message : String(error),
                }).catch(() => undefined)
            }
            appendTranscript({
                kind: 'error',
                text: error instanceof Error ? error.message : String(error),
            })
            setStage('select-directory')
            loggerRef.current?.log('stage_change', { stage: 'select-directory' })
        }
    }, [appendTranscript, flushAndExit, mode, runDirectory, text])

    const interruptPendingGlossarySelection = useCallback(async () => {
        const pendingGlossarySelection = pendingGlossarySelectionRef.current

        if (!shouldInterruptPendingGlossarySelection(pendingGlossarySelection)) {
            return
        }

        const selectionToInterrupt = pendingGlossarySelection

        try {
            await updateTask(runDirectory, selectionToInterrupt.task.task_code, { status: 'interrupted' })
            pendingGlossarySelectionRef.current = null
        } catch (error) {
            loggerRef.current?.log('pending_glossary_selection_interrupt_failed', {
                taskCode: selectionToInterrupt.task.task_code,
                error: error instanceof Error ? error.message : String(error),
            })
        }
    }, [runDirectory])

    const returnToStartMenu = useCallback(async () => {
        await interruptPendingGlossarySelection()
        loggerRef.current?.close()
        loggerRef.current = null
        effectiveConfigRef.current = null
        forcedResumeConfigRef.current = null
        setAgent(null)
        setMessages([])
        setInput('')
        setPendingGlossarySelection(null)
        setPendingConfigResume(null)
        setHoveredConfigMismatchChoice(null)
        setConfigMismatchKeyboardChoice(null)
        setHoveredTaskCompleteTarget(null)
        setHoveredTaskSelectionTarget(null)
        setTaskSelectionKeyboardTarget(null)
        setSettingsEdit(null)
        setHoveredSettingsEditTarget(null)
        setSettingsEditKeyboardTarget(null)
        setHoveredSetupWizardTarget(null)
        setStage('start-menu')
        setTranscript([
            {
                id: 'start',
                kind: 'system',
                text: text.message.projectReady(runDirectory),
            },
        ])
    }, [interruptPendingGlossarySelection, runDirectory, text])

    const openSettings = useCallback(async () => {
        setStage('settings')
        setInput('')
        setSettingsLoadError(null)
        setSettingsSaveMessage(null)
        setSettingsEdit(null)
        setHoveredSettingsEditTarget(null)
        setSettingsEditKeyboardTarget(null)
        setSettingsEditError(null)
        setHoveredSettingsTarget(null)
        setSettingsEditReturnStage('settings')

        try {
            const editableConfig = await loadEditableAgentConfig()
            const preferences = await loadPreferences()
            setSettingsConfig(editableConfig.config)
            setSettingsSavedConfig(editableConfig.config)
            setSettingsPreferences(preferences)
            setSettingsSavedPreferences(preferences)
            setSelectedSettingsCategoryIndex(0)
            setSelectedSettingsFieldIndex(0)
            setSettingsFocusArea('categories')
            setSelectedSettingsButtonIndex(0)
        } catch (error) {
            setSettingsLoadError(error instanceof Error ? error.message : String(error))
        }
    }, [])

    const openSetupWizard = useCallback(async (firstRun: boolean) => {
        setStage('setup-wizard')
        setInput('')
        setSettingsLoadError(null)
        setSettingsSaveMessage(null)
        setSettingsEdit(null)
        setHoveredSettingsEditTarget(null)
        setSettingsEditKeyboardTarget(null)
        setSettingsEditError(null)
        setHoveredSettingsTarget(null)
        setHoveredSetupWizardTarget(null)
        setSettingsEditReturnStage('setup-wizard')
        setSelectedSetupWizardStepIndex(0)
        setSelectedSetupWizardFieldIndex(0)
        setSetupWizardFocusArea('buttons')
        setSetupWizardButtonFocus('primary')
        setSetupWizardFirstRun(firstRun)
        setSetupWizardApiTest({ status: 'idle' })
        setupWizardApiTestRunRef.current += 1

        try {
            const editableConfig = await loadEditableAgentConfig()
            const preferences = await loadPreferences()
            setSettingsConfig(editableConfig.config)
            setSettingsSavedConfig(editableConfig.config)
            setSettingsPreferences(preferences)
            setSettingsSavedPreferences(preferences)
            setTuiLanguage(preferences.tuiLanguage)
        } catch (error) {
            setSettingsLoadError(error instanceof Error ? error.message : String(error))
        }
    }, [])

    const startMenuChoice = useCallback((choice: StartMenuChoice) => {
        setHoveredStartMenuChoice(null)

        if (choice === 'settings') {
            void openSettings()
            return
        }

        if (choice === 'setupWizard') {
            void openSetupWizard(false)
            return
        }

        setMode(choice)
        void selectDirectory(defaultDirectory, choice)
    }, [openSettings, openSetupWizard, selectDirectory])

    const runSelectedTask = useCallback(async (
        task: ProjectTask,
        config: AgentConfig,
        resumed: boolean,
        glossarySource: TaskGlossarySource|null,
    ) => {
        try {
            const taskContext = createTaskRunContext(runDirectory, task.task_code, glossarySource)
            effectiveConfigRef.current = config
            pendingGlossarySelectionRef.current = null
            setPendingConfigResume(null)
            setPendingGlossarySelection(null)
            setActiveTask(task)
            setWorkerPanes(initialWorkerPanes(mode === 'translation' ? config.translationWorkerParallelism : config.glossaryWorkerParallelism))

            const logger = await createRunLogger(runDirectory, createTaskRunPaths(runDirectory, task.task_code).logDirectory, {
                enableRunLogs: config.enableRunLogs,
                enableDebugLogs: config.enableDebugLogs,
            })
            loggerRef.current = logger
            logger.log('stage_change', { stage: 'initializing' })
            logger.log('task_selected', {
                taskCode: task.task_code,
                resumed,
                glossarySource,
            })
            logger.log('task_config_selected', {
                taskCode: task.task_code,
                source: resumed ? 'task_snapshot_or_current' : 'current',
                differences: resumed ? compareTaskConfigSnapshot(config, task.config_snapshot) : [],
            })
            logger.log('config_loaded', {
                model: config.model,
                contextWindow: config.contextWindow,
                manualTransFile: config.manualTransFile,
                reasoningEffort: config.reasoningEffort,
                glossaryWorkerParallelism: config.glossaryWorkerParallelism,
                glossaryWorkerBatchSize: config.glossaryWorkerBatchSize,
                glossaryWorkerRecursionLimit: config.glossaryWorkerRecursionLimit,
                glossaryReviewEveryCompletedBatches: config.glossaryReviewEveryCompletedBatches,
                glossaryReviewRecursionLimit: config.glossaryReviewRecursionLimit,
                translationWorkerParallelism: config.translationWorkerParallelism,
                translationWorkerBatchSize: config.translationWorkerBatchSize,
                translationWorkerRecursionLimit: config.translationWorkerRecursionLimit,
                apiRetryAttempts: config.apiRetryAttempts,
                mode,
                taskCode: task.task_code,
                glossarySource,
            })
            await updateTask(runDirectory, task.task_code, {
                stage: 'initializing',
                status: 'running',
                glossary_source: glossarySource,
                log_files: createTaskLogFilesUpdate(runDirectory, logger),
                completed_at: null,
                last_error: null,
            })

            const initialTranscript: TranscriptEntry[] = [
                {
                    id: 'ready',
                    kind: 'system',
                    text: text.message.runDirectory(runDirectory),
                },
                {
                    id: 'task-code',
                    kind: 'system',
                    text: text.message.taskLine(task.task_code, resumed),
                },
                ...(mode === 'translation' ? [{
                    id: 'glossary-source',
                    kind: 'system' as const,
                    text: text.message.glossaryLine(glossarySource?.task_code ?? null),
                }] : []),
                {
                    id: 'task-directory',
                    kind: 'system',
                    text: text.message.outputLine(createTaskRunPaths(runDirectory, task.task_code).taskDirectory),
                },
                ...(logger.filePath ? [{
                    id: 'log-file',
                    kind: 'system' as const,
                    text: text.message.runLogLine(logger.filePath),
                }] : []),
                ...(logger.debugFilePath ? [{
                    id: 'debug-log-file',
                    kind: 'system' as const,
                    text: text.message.debugLogLine(logger.debugFilePath),
                }] : []),
                {
                    id: 'preflight',
                    kind: 'system',
                    text: mode === 'translation'
                        ? text.message.translationPreflightStart
                        : text.message.glossaryPreflightStart,
                },
            ]
            initialTranscript.forEach(entry => logger.log('transcript', entry))
            setTranscript(initialTranscript)
            setStage('preflighting')
            logger.log('stage_change', { stage: 'preflighting' })

            if (mode === 'translation') {
                let translationPreflight: TranslationPreflight

                try {
                    const preflight = await runTranslationPreflight(runDirectory, handleToolEvent, handleApiEvent, taskContext, config)
                    translationPreflight = preflight.preflight
                    logger.log('translation_preflight_result', {
                        preflight: preflight.preflight,
                        answer: preflight.answer,
                        messages: preflight.messages,
                    })
                    appendTranscript({
                        kind: 'assistant',
                        text: formatTranslationPreflight(preflight.preflight, text),
                    })
                } catch (error) {
                    logger.log('translation_preflight_failed', {
                        error: error instanceof Error ? error.message : String(error),
                    })
                    appendTranscript({
                        kind: 'error',
                        text: text.message.translationPreflightFailed(error instanceof Error ? error.message : String(error)),
                    })
                    await updateTask(runDirectory, task.task_code, {
                        status: 'failed',
                        last_error: error instanceof Error ? error.message : String(error),
                    })
                    setStage('select-directory')
                    logger.log('stage_change', { stage: 'select-directory' })
                    return
                }

                appendTranscript({
                    kind: 'system',
                    text: text.message.translationWorkersStart,
                })
                setStage('extracting')
                logger.log('stage_change', { stage: 'extracting' })

                const workerResult = await runTranslationWorkers(runDirectory, translationPreflight, handleToolEvent, handleWorkerUpdate, handleApiEvent, taskContext, config)
                logger.log('translation_worker_stage_result', workerResult)

                appendTranscript({
                    kind: workerResult.failedBatches > 0 ? 'error' : 'system',
                    text: text.message.translationWorkerFinished(workerResult.completedBatches, workerResult.totalBatches, workerResult.failedBatches, workerResult.restoredBatches),
                })

                if (workerResult.failedBatches > 0) {
                    await updateTask(runDirectory, task.task_code, {
                        status: 'failed',
                        last_error: text.message.translationWorkerFailed(workerResult.failedBatches),
                    })
                    setStage('select-directory')
                    logger.log('stage_change', { stage: 'select-directory' })
                    return
                }

                setStage('exporting')
                logger.log('stage_change', { stage: 'exporting' })
                appendTranscript({
                    kind: 'system',
                    text: text.message.exportingManualTrans,
                })

                const exportResult = await exportTranslationResult(runDirectory, taskContext, config)
                logger.log('translation_export_result', exportResult)
                appendTranscript({
                    kind: 'system',
                    text: formatTranslationExportResult(exportResult, text),
                })
            } else {
                let preflightPlan: GlossaryPlan

                try {
                    const preflight = await runGlossaryPreflight(runDirectory, handleToolEvent, handleApiEvent, taskContext, config)
                    preflightPlan = preflight.plan
                    logger.log('preflight_result', {
                        plan: preflight.plan,
                        answer: preflight.answer,
                        messages: preflight.messages,
                    })
                    appendTranscript({
                        kind: 'assistant',
                        text: formatGlossaryPlan(preflight.plan, text),
                    })
                } catch (error) {
                    logger.log('preflight_failed', {
                        error: error instanceof Error ? error.message : String(error),
                    })
                    appendTranscript({
                        kind: 'error',
                        text: text.message.glossaryPreflightFailed(error instanceof Error ? error.message : String(error)),
                    })
                    await updateTask(runDirectory, task.task_code, {
                        status: 'failed',
                        last_error: error instanceof Error ? error.message : String(error),
                    })
                    setStage('select-directory')
                    logger.log('stage_change', { stage: 'select-directory' })
                    return
                }

                appendTranscript({
                    kind: 'system',
                    text: text.message.glossaryWorkersStart,
                })
                setStage('extracting')
                logger.log('stage_change', { stage: 'extracting' })

                const workerResult = await runGlossaryWorkers(runDirectory, preflightPlan, handleToolEvent, handleWorkerUpdate, handleReviewUpdate, handleApiEvent, taskContext, config)
                await flushGlossaryState(taskContext.glossaryStateRoot)
                logger.log('worker_stage_result', workerResult)

                appendTranscript({
                    kind: workerResult.failedBatches > 0 || workerResult.failedReviews > 0 ? 'error' : 'system',
                    text: text.message.glossaryWorkerFinished(workerResult.completedBatches, workerResult.totalBatches, workerResult.failedBatches, workerResult.completedReviews, workerResult.failedReviews),
                })

                if (workerResult.failedBatches > 0 || workerResult.failedReviews > 0) {
                    await updateTask(runDirectory, task.task_code, {
                        status: 'failed',
                        last_error: text.message.glossaryWorkerFailed(workerResult.failedBatches, workerResult.failedReviews),
                    })
                    setStage('select-directory')
                    logger.log('stage_change', { stage: 'select-directory' })
                    return
                }
            }

            appendTranscript({
                kind: 'system',
                text: mode === 'translation'
                    ? text.message.translationReady
                    : text.message.glossaryReady,
            })
            setHoveredTaskCompleteTarget(null)
            setStage('task-complete')
            await updateTask(runDirectory, task.task_code, {
                stage: 'ready',
                status: 'completed',
                completed_at: new Date().toISOString(),
            })
            logger.log('stage_change', { stage: 'task-complete' })
        } catch (error) {
            loggerRef.current?.log('initialization_failed', {
                error: error instanceof Error ? error.message : String(error),
            })
            if (task) {
                await updateTask(runDirectory, task.task_code, {
                    status: 'failed',
                    last_error: error instanceof Error ? error.message : String(error),
                }).catch(() => undefined)
            }
            appendTranscript({
                kind: 'error',
                text: error instanceof Error ? error.message : String(error),
            })
            setStage('select-directory')
            loggerRef.current?.log('stage_change', { stage: 'select-directory' })
        }
    }, [appendTranscript, handleApiEvent, handleReviewUpdate, handleToolEvent, handleWorkerUpdate, mode, runDirectory, text])
    runSelectedTaskRef.current = runSelectedTask

    // Currently dead code: Ask mode is disabled; kept for possible future reuse.
    const submitPrompt = useCallback(async (prompt: string) => {
        if (!agent) {
            return
        }

        const trimmedPrompt = prompt.trim()

        if (trimmedPrompt.length === 0) {
            return
        }

        if (trimmedPrompt === '/dir') {
            loggerRef.current?.log('directory_change_requested', { currentDirectory: runDirectory })
            loggerRef.current?.close()
            loggerRef.current = null
            effectiveConfigRef.current = null
            setAgent(null)
            setMessages([])
            setInput('')
            setStage('select-directory')
            appendTranscript({
                kind: 'system',
                text: text.message.chooseRunDirectory(runDirectory),
            })
            return
        }

        setInput('')
        setStage('running')
        loggerRef.current?.log('stage_change', { stage: 'running' })
        appendTranscript({
            kind: 'user',
            text: trimmedPrompt,
        })

        const nextMessages: BaseMessageLike[] = [
            ...messages,
            {
                role: 'user',
                content: trimmedPrompt,
            },
        ]

        try {
            loggerRef.current?.log('ask_invocation_started', {
                prompt: trimmedPrompt,
                messages: nextMessages,
            })
            const result = await invokeProjectAgent(agent, {
                messages: nextMessages,
            }, undefined, handleApiEvent, effectiveConfigRef.current ?? undefined)
            const resultMessages = result.messages
            const answer = getLastMessageContent(resultMessages)

            setMessages(resultMessages)
            loggerRef.current?.log('ask_invocation_completed', {
                answer,
                messages: resultMessages,
            })
            appendTranscript({
                kind: 'assistant',
                text: answer || text.message.emptyResponse,
            })
            setStage('ready')
            loggerRef.current?.log('stage_change', { stage: 'ready' })
        } catch (error) {
            loggerRef.current?.log('ask_invocation_failed', {
                error: error instanceof Error ? error.message : String(error),
            })
            appendTranscript({
                kind: 'error',
                text: error instanceof Error ? error.message : String(error),
            })
            setStage('ready')
            loggerRef.current?.log('stage_change', { stage: 'ready' })
        }
    }, [agent, appendTranscript, handleApiEvent, messages, runDirectory, text])

    const showReviewPane = mode === 'glossary' && stage === 'extracting' && reviewPane.status !== 'idle' && reviewPane.status !== 'completed'
    const showWorkerPanes = stage === 'extracting' && !showReviewPane
    const showPane = showReviewPane || showWorkerPanes
    const fixedRows = 5
    const contentRows = Math.max(1, rows - fixedRows)
    const transcriptViewportRows = contentRows
    const dualPaneLayout = useMemo(() => calculateDualPaneLayout(columns, contentRows), [columns, contentRows])
    const paneViewportRows = contentRows
    const workerPaneGridLayout = useMemo(
        () => calculateScrollableWorkerPaneGridLayout(workerPanes, dualPaneLayout.paneWidth, paneViewportRows),
        [dualPaneLayout.paneWidth, paneViewportRows, workerPanes],
    )
    const paneContentRows = showReviewPane
        ? getReviewPaneContentRows(reviewPane, undefined, text)
        : showWorkerPanes
            ? workerPaneGridLayout.contentRows
            : 0
    const workerPaneContentRows = useMemo(() => {
        const rowsBySlot = new Map<number, number>()

        for (const pane of workerPanes) {
            rowsBySlot.set(pane.slotIndex, formatWorkerPaneLines(pane, workerPaneGridLayout.paneWidth, text).length)
        }

        return rowsBySlot
    }, [text, workerPaneGridLayout.paneWidth, workerPanes])
    const workerPaneViewportRows = getWorkerPaneViewportRows()
    const reviewPaneViewportRows = Math.max(1, contentRows - 2)
    const reviewPaneContentRows = getReviewPaneContentRows(reviewPane, dualPaneLayout.paneWidth, text)
    const transcriptLines = useMemo(() => flattenTranscriptEntries(transcript, text), [text, transcript])
    const maxScrollOffset = getMaxScrollOffset(transcriptLines.length, transcriptViewportRows)
    const visibleTaskChoices = useMemo(() => getVisibleTaskChoices(taskChoices), [taskChoices])
    const visibleGlossaryChoices = useMemo(() => getVisibleTaskChoices(glossaryChoices), [glossaryChoices])
    const currentSettingsCategory = settingsCategories[selectedSettingsCategoryIndex] ?? settingsCategories[0]!
    const currentSettingsFields = currentSettingsCategory.fields
    const selectedSettingsField = currentSettingsFields[Math.min(selectedSettingsFieldIndex, Math.max(0, currentSettingsFields.length - 1))]
    const currentSetupWizardStep = SETUP_WIZARD_STEPS[selectedSetupWizardStepIndex] ?? SETUP_WIZARD_STEPS[0]!
    const currentSetupWizardFields = currentSetupWizardStep.fields
    const selectedSetupWizardField = currentSetupWizardFields[Math.min(selectedSetupWizardFieldIndex, Math.max(0, currentSetupWizardFields.length - 1))]
    const startMenuLayout = useMemo(
        () => calculateStartMenuLayout({
            height: contentRows,
            text,
            terminalColumns: columns,
        }),
        [columns, contentRows, text],
    )
    const configMismatchLayout = useMemo(
        () => calculateConfigMismatchLayout({
            differenceCount: pendingConfigResume?.differences.length ?? 0,
            height: contentRows,
            text,
            terminalColumns: columns,
        }),
        [columns, contentRows, pendingConfigResume?.differences.length, text],
    )
    const taskSelectionLayout = useMemo(
        () => calculateTaskSelectionDialogLayout({
            height: contentRows,
            taskCount: stage === 'select-glossary' ? visibleGlossaryChoices.length : visibleTaskChoices.length,
            text,
            terminalColumns: columns,
        }),
        [columns, contentRows, stage, text, visibleGlossaryChoices.length, visibleTaskChoices.length],
    )
    const taskCompleteLayout = useMemo(
        () => calculateTaskCompleteLayout({
            height: contentRows,
            text,
            terminalColumns: columns,
        }),
        [columns, contentRows, text],
    )
    const settingsLayout = useMemo(
        () => calculateSettingsLayout({
            categoryCount: settingsCategories.length,
            fieldCount: maxSettingsFieldCount,
            fieldHitCount: currentSettingsFields.length,
            height: contentRows,
            text,
            terminalColumns: columns,
        }),
        [columns, contentRows, currentSettingsFields.length, text],
    )
    const settingsEditLayout = useMemo(
        () => calculateSettingsEditLayout({
            height: contentRows,
            optionCount: settingsEdit ? getSettingsEditOptions(settingsFieldDefinitions.get(settingsEdit.field)?.kind ?? 'string').length : 0,
            text,
            terminalColumns: columns,
        }),
        [columns, contentRows, settingsEdit, text],
    )
    const setupWizardLayout = useMemo(
        () => calculateSetupWizardLayout({
            fieldCount: currentSetupWizardFields.length,
            height: contentRows,
            selectedStepIndex: selectedSetupWizardStepIndex,
            text,
            terminalColumns: columns,
        }),
        [columns, contentRows, currentSetupWizardFields.length, selectedSetupWizardStepIndex, text],
    )
    const displayedSettingsEditTarget = settingsEditKeyboardTarget ?? hoveredSettingsEditTarget
    const displayedTaskSelectionTarget = taskSelectionKeyboardTarget ?? hoveredTaskSelectionTarget
    const displayedConfigMismatchChoice = configMismatchKeyboardChoice ?? hoveredConfigMismatchChoice

    const absorbModalMouseWheelValue = useCallback((value: string): boolean => {
        if (!isModalStage(stage) || parseMouseWheelEvents(value).length === 0) {
            return false
        }

        lastRawMouseWheelAtRef.current = Date.now()
        return true
    }, [stage])

    const scrollBy = useCallback((delta: number) => {
        setScrollOffsetFromBottom(current => clampScrollOffset(current + delta, transcriptLines.length, transcriptViewportRows))
    }, [transcriptLines.length, transcriptViewportRows])

    useEffect(() => {
        if (startupSetupCheckStartedRef.current) {
            return
        }

        startupSetupCheckStartedRef.current = true
        let cancelled = false

        void (async () => {
            try {
                const preferences = await loadPreferences()

                if (cancelled) {
                    return
                }

                setTuiLanguage(preferences.tuiLanguage)

                if (!preferences.setupWizardCompleted) {
                    void openSetupWizard(true)
                } else {
                    setStage('start-menu')
                }
            } catch (error) {
                if (!cancelled) {
                    appendTranscript({
                        kind: 'error',
                        text: error instanceof Error ? error.message : String(error),
                    })
                    setStage('start-menu')
                }
            }
        })()

        return () => {
            cancelled = true
        }
    }, [appendTranscript, openSetupWizard])

    const scrollPaneBy = useCallback((delta: number) => {
        setPaneScrollOffsetFromTop(current => clampTopScrollOffset(current + delta, paneContentRows, paneViewportRows))
    }, [paneContentRows, paneViewportRows])

    const canScrollTopViewport = useCallback((offset: number, delta: number, lineCount: number, viewportRows: number): boolean => {
        const nextOffset = clampTopScrollOffset(offset + delta, lineCount, viewportRows)

        return nextOffset !== offset
    }, [])

    const scrollWorkerPaneBy = useCallback((slotIndex: number, delta: number) => {
        const contentRowCount = workerPaneContentRows.get(slotIndex) ?? 0

        setWorkerPaneScrollOffsets(current => ({
            ...current,
            [slotIndex]: clampTopScrollOffset((current[slotIndex] ?? 0) + delta, contentRowCount, workerPaneViewportRows),
        }))
    }, [workerPaneContentRows, workerPaneViewportRows])

    const scrollReviewPaneBy = useCallback((delta: number) => {
        setReviewPaneScrollOffset(current => clampTopScrollOffset(current + delta, reviewPaneContentRows, reviewPaneViewportRows))
    }, [reviewPaneContentRows, reviewPaneViewportRows])

    const changeToDirectorySelection = useCallback(async () => {
        await interruptPendingGlossarySelection()
        setInput('')
        setStage('select-directory')
        setPendingConfigResume(null)
        setHoveredConfigMismatchChoice(null)
        setConfigMismatchKeyboardChoice(null)
        setHoveredTaskCompleteTarget(null)
        forcedResumeConfigRef.current = null
        setHoveredTaskSelectionTarget(null)
        setTaskSelectionKeyboardTarget(null)
        appendTranscript({
            kind: 'system',
            text: text.message.chooseRunDirectory(runDirectory),
        })
    }, [appendTranscript, interruptPendingGlossarySelection, runDirectory, text])

    const startSelectedTask = useCallback((selectedIndex: number) => {
        void initializeAgent(getSelectedTaskCode(visibleTaskChoices, selectedIndex))
    }, [initializeAgent, visibleTaskChoices])

    const startSelectedGlossary = useCallback((selectedIndex: number) => {
        if (!pendingGlossarySelection) {
            return
        }

        const selectedGlossaryTask = getSelectedTaskCode(visibleGlossaryChoices, selectedIndex)
            ? visibleGlossaryChoices[selectedIndex] ?? null
            : null
        const glossarySource = selectedGlossaryTask ? createGlossarySourceFromTask(selectedGlossaryTask) : null

        pendingGlossarySelectionRef.current = null
        void runSelectedTask(
            pendingGlossarySelection.task,
            pendingGlossarySelection.config,
            pendingGlossarySelection.resumed,
            glossarySource,
        )
    }, [pendingGlossarySelection, runSelectedTask, visibleGlossaryChoices])

    const continueWithTaskConfig = useCallback(() => {
        if (!pendingConfigResume) {
            return
        }

        forcedResumeConfigRef.current = {
            taskCode: pendingConfigResume.task.task_code,
            config: pendingConfigResume.taskConfig,
        }
        setPendingConfigResume(null)
        setHoveredConfigMismatchChoice(null)
        setConfigMismatchKeyboardChoice(null)
        void initializeAgent(pendingConfigResume.task.task_code)
    }, [initializeAgent, pendingConfigResume])

    const backFromConfigMismatch = useCallback(() => {
        setPendingConfigResume(null)
        setHoveredConfigMismatchChoice(null)
        setConfigMismatchKeyboardChoice(null)
        forcedResumeConfigRef.current = null
        setStage('select-task')
    }, [])

    const selectConfigMismatchChoice = useCallback((choice: ConfigMismatchChoice) => {
        if (choice === 'use-task-config') {
            continueWithTaskConfig()
            return
        }

        backFromConfigMismatch()
    }, [backFromConfigMismatch, continueWithTaskConfig])

    const handleConfigMismatchMouseValue = useCallback((value: string): boolean => {
        if (stage !== 'confirm-config') {
            return false
        }

        if (absorbModalMouseWheelValue(value)) {
            return true
        }

        const events = parseMouseEvents(value)
        let handled = false

        for (const event of events) {
            if (event.kind === 'wheel') {
                continue
            }

            setConfigMismatchKeyboardChoice(null)
            const target = getConfigMismatchMouseTarget(event, configMismatchLayout)

            if (!target) {
                setHoveredConfigMismatchChoice(null)
                handled = true
                continue
            }

            setHoveredConfigMismatchChoice(target.choice)
            handled = true

            if (event.kind === 'press' && event.button === 'left') {
                selectConfigMismatchChoice(target.choice)
            }
        }

        return handled
    }, [absorbModalMouseWheelValue, configMismatchLayout, selectConfigMismatchChoice, stage])

    const handleTaskCompleteMouseValue = useCallback((value: string): boolean => {
        if (stage !== 'task-complete') {
            return false
        }

        if (absorbModalMouseWheelValue(value)) {
            return true
        }

        const events = parseMouseEvents(value)
        let handled = false

        for (const event of events) {
            if (event.kind === 'wheel') {
                continue
            }

            const target = getTaskCompleteMouseTarget(event, taskCompleteLayout)

            if (!target) {
                setHoveredTaskCompleteTarget(null)
                handled = true
                continue
            }

            setHoveredTaskCompleteTarget(target)
            handled = true

            if (event.kind === 'press' && event.button === 'left') {
                void returnToStartMenu()
            }
        }

        return handled
    }, [absorbModalMouseWheelValue, returnToStartMenu, stage, taskCompleteLayout])

    const openSettingsFieldEditor = useCallback((field: SettingsFieldId, returnStage: SettingsEditReturnStage = 'settings') => {
        const definition = settingsFieldDefinitions.get(field)
        const currentValue = getSettingsFieldRawValue(field, settingsConfig, settingsPreferences)
        const options = getSettingsEditOptions(definition?.kind ?? 'string')
        const selectedOptionIndex = Math.max(0, options.findIndex(option => option === String(currentValue)))

        setSettingsEditReturnStage(returnStage)
        setSettingsEdit({
            field,
            input: definition?.kind === 'secret' ? '' : formatSettingsRawValue(currentValue),
            selectedOptionIndex: selectedOptionIndex === -1 ? 0 : selectedOptionIndex,
        })
        setHoveredSettingsEditTarget(null)
        setSettingsEditKeyboardTarget(null)
        setSettingsEditError(null)
        setStage('settings-edit')
    }, [settingsConfig, settingsPreferences])

    const cancelSettingsEdit = useCallback(() => {
        setSettingsEdit(null)
        setHoveredSettingsEditTarget(null)
        setSettingsEditKeyboardTarget(null)
        setSettingsEditError(null)
        setStage(settingsEditReturnStage)
    }, [settingsEditReturnStage])

    const pasteIntoSettingsEdit = useCallback(async () => {
        const pasteContext = clipboardPasteContextRef.current
        const pastedText = await readClipboardText()

        if (!isCurrentClipboardPasteContext(clipboardPasteContextRef.current, pasteContext)) {
            return
        }

        if (!pastedText) {
            showClipboardMessage(text.message.clipboardUnavailable)
            return
        }

        setHoveredSettingsEditTarget(null)
        setSettingsEdit(current => current ? { ...current, input: `${current.input}${pastedText}` } : current)
        setSettingsEditError(null)
        setClipboardMessage(null)
    }, [showClipboardMessage, text])

    const pasteIntoPromptInput = useCallback(async () => {
        const pasteContext = clipboardPasteContextRef.current
        const pastedText = await readClipboardText()

        if (!isCurrentClipboardPasteContext(clipboardPasteContextRef.current, pasteContext)) {
            return
        }

        if (!pastedText) {
            showClipboardMessage(text.message.clipboardUnavailable)
            return
        }

        setInput(current => `${current}${pastedText}`)
        setClipboardMessage(null)
    }, [showClipboardMessage, text])

    const applySettingsEdit = useCallback(() => {
        if (!settingsEdit) {
            return
        }

        const definition = settingsFieldDefinitions.get(settingsEdit.field)

        if (!definition) {
            return
        }

        const currentValue = getSettingsFieldRawValue(settingsEdit.field, settingsConfig, settingsPreferences)
        const parsedValue = parseSettingsEditValue(definition, settingsEdit, currentValue, text)

        if (!parsedValue.ok) {
            setSettingsEditError(parsedValue.error)
            return
        }

        if (settingsEdit.field === 'tuiLanguage') {
            setSettingsPreferences(current => ({
                ...current,
                tuiLanguage: parsedValue.value as TuiLanguage,
            }))
        } else {
            setSettingsConfig(current => ({
                ...current,
                [settingsEdit.field]: parsedValue.value,
            }))
        }
        setSettingsSaveMessage(null)
        cancelSettingsEdit()
    }, [cancelSettingsEdit, settingsConfig, settingsEdit, settingsPreferences, text])

    const saveSettings = useCallback(async () => {
        try {
            await saveAgentConfig(settingsConfig)
            await savePreferences(settingsPreferences)
            setSettingsSavedConfig(settingsConfig)
            setSettingsSavedPreferences(settingsPreferences)
            setTuiLanguage(settingsPreferences.tuiLanguage)
            setSettingsSaveMessage(text.settings.saved)
            setSettingsLoadError(null)
        } catch (error) {
            setSettingsSaveMessage(error instanceof Error ? error.message : String(error))
        }
    }, [settingsConfig, settingsPreferences, text])

    const cancelSettings = useCallback(() => {
        setSettingsConfig(settingsSavedConfig)
        setSettingsPreferences(settingsSavedPreferences)
        setSettingsSaveMessage(null)
        void returnToStartMenu()
    }, [returnToStartMenu, settingsSavedConfig, settingsSavedPreferences])

    const selectSettingsButton = useCallback((button: SettingsButton) => {
        if (button === 'save') {
            void saveSettings()
            return
        }

        cancelSettings()
    }, [cancelSettings, saveSettings])

    const runSetupWizardApiTest = useCallback(async () => {
        const runId = setupWizardApiTestRunRef.current + 1
        setupWizardApiTestRunRef.current = runId
        setSetupWizardApiTest({ status: 'running' })
        setSettingsSaveMessage(text.setupWizard.apiTestRunning)

        try {
            const response = await testConfiguredModel(settingsConfig, handleApiEvent)

            if (setupWizardApiTestRunRef.current !== runId) {
                return
            }

            setSetupWizardApiTest({
                status: 'succeeded',
                response,
            })
            setSettingsSaveMessage(text.setupWizard.apiTestSucceeded)
        } catch (error) {
            if (setupWizardApiTestRunRef.current !== runId) {
                return
            }

            setSetupWizardApiTest({
                status: 'failed',
                error: error instanceof Error ? error.message : String(error),
            })
            setSettingsSaveMessage(text.setupWizard.apiTestFailed)
        }
    }, [handleApiEvent, settingsConfig, text])

    const goToSetupWizardStep = useCallback((stepIndex: number) => {
        const nextStepIndex = clampIndex(stepIndex, SETUP_WIZARD_STEPS.length)
        const nextFieldCount = SETUP_WIZARD_STEPS[nextStepIndex]?.fields.length ?? 0

        setSelectedSetupWizardStepIndex(nextStepIndex)
        setSelectedSetupWizardFieldIndex(0)
        setSetupWizardFocusArea(nextFieldCount > 0 ? 'fields' : 'buttons')
        setSetupWizardButtonFocus('primary')
        setHoveredSetupWizardTarget(null)
        setSettingsSaveMessage(null)
    }, [])

    const selectSetupWizardButton = useCallback(async (button: SetupWizardButton) => {
        if (button === 'previous') {
            setupWizardApiTestRunRef.current += 1
            goToSetupWizardStep(selectedSetupWizardStepIndex - 1)
            return
        }

        if (button === 'retest') {
            void runSetupWizardApiTest()
            return
        }

        if (button === 'next') {
            const nextStepIndex = clampIndex(selectedSetupWizardStepIndex + 1, SETUP_WIZARD_STEPS.length)

            if (SETUP_WIZARD_STEPS[nextStepIndex]?.id === 'apiTest') {
                goToSetupWizardStep(nextStepIndex)
                void runSetupWizardApiTest()
            } else {
                setupWizardApiTestRunRef.current += 1
                goToSetupWizardStep(nextStepIndex)
            }
            return
        }

        if (button === 'skipRecommended') {
            setupWizardApiTestRunRef.current += 1
            goToSetupWizardStep(selectedSetupWizardStepIndex + 1)
            return
        }

        if (button === 'applyRecommended') {
            try {
                setupWizardApiTestRunRef.current += 1
                const recommendedConfig = getRecommendedAgentConfig()
                const nextConfig = applyRecommendedAgentConfig(settingsConfig, recommendedConfig)

                setSettingsConfig(nextConfig)
                goToSetupWizardStep(selectedSetupWizardStepIndex + 1)
            } catch (error) {
                setSettingsSaveMessage(error instanceof Error ? error.message : String(error))
            }
            return
        }

        try {
            setupWizardApiTestRunRef.current += 1
            const completedPreferences: Preferences = {
                ...settingsPreferences,
                setupWizardCompleted: true,
            }

            await saveAgentConfig(settingsConfig)
            await savePreferences(completedPreferences)
            setSettingsSavedConfig(settingsConfig)
            setSettingsPreferences(completedPreferences)
            setSettingsSavedPreferences(completedPreferences)
            setTuiLanguage(completedPreferences.tuiLanguage)
            setSettingsSaveMessage(text.setupWizard.saved)
            setSettingsLoadError(null)
            setSetupWizardFirstRun(false)
            void returnToStartMenu()
        } catch (error) {
            setSettingsSaveMessage(error instanceof Error ? error.message : String(error))
        }
    }, [goToSetupWizardStep, returnToStartMenu, runSetupWizardApiTest, selectedSetupWizardStepIndex, settingsConfig, settingsPreferences, text])

    const cancelSetupWizard = useCallback(() => {
        setupWizardApiTestRunRef.current += 1
        setSettingsConfig(settingsSavedConfig)
        setSettingsPreferences(settingsSavedPreferences)
        setSetupWizardApiTest({ status: 'idle' })
        setSettingsSaveMessage(null)

        if (!setupWizardFirstRun) {
            void returnToStartMenu()
        }
    }, [returnToStartMenu, settingsSavedConfig, settingsSavedPreferences, setupWizardFirstRun])

    const handleEscape = useCallback(() => {
        if (stage === 'settings-edit' && settingsEdit) {
            cancelSettingsEdit()
            return
        }

        if (stage === 'settings') {
            cancelSettings()
            return
        }

        if (stage === 'setup-wizard') {
            cancelSetupWizard()
            return
        }

        if (stage === 'select-task' || stage === 'select-glossary') {
            void returnToStartMenu()
            return
        }

        if (stage === 'confirm-config') {
            backFromConfigMismatch()
            return
        }

        if (stage === 'select-directory') {
            void returnToStartMenu()
            return
        }

        if (stage === 'task-complete') {
            void returnToStartMenu()
            return
        }

        // Currently dead code: Ask mode is disabled; kept for possible future reuse.
        if (stage === 'ready' && input.length > 0) {
            setInput('')
            return
        }

        appendTranscript({
            kind: 'system',
            text: 'Press Ctrl+C twice to exit.',
        })
    }, [appendTranscript, backFromConfigMismatch, cancelSettings, cancelSettingsEdit, cancelSetupWizard, input.length, returnToStartMenu, settingsEdit, stage])

    const handleStartMenuMouseValue = useCallback((value: string): boolean => {
        if (stage !== 'start-menu') {
            return false
        }

        if (absorbModalMouseWheelValue(value)) {
            return true
        }

        const events = parseMouseEvents(value)
        let handled = false

        for (const event of events) {
            if (event.kind === 'wheel') {
                continue
            }

            const target = getStartMenuMouseTarget(event, startMenuLayout)

            if (!target) {
                setHoveredStartMenuChoice(null)
                handled = true
                continue
            }

            setHoveredStartMenuChoice(target.choice)
            setSelectedStartMenuIndex(getStartMenuChoiceIndex(target.choice))
            handled = true

            if (event.kind === 'press' && event.button === 'left') {
                startMenuChoice(target.choice)
            }
        }

        return handled
    }, [absorbModalMouseWheelValue, stage, startMenuChoice, startMenuLayout])

    const handleSettingsMouseValue = useCallback((value: string): boolean => {
        if (stage !== 'settings') {
            return false
        }

        if (absorbModalMouseWheelValue(value)) {
            return true
        }

        const events = parseMouseEvents(value)
        let handled = false

        for (const event of events) {
            if (event.kind === 'wheel') {
                continue
            }

            const target = getSettingsMouseTarget(event, settingsLayout)

            if (!target) {
                setHoveredSettingsTarget(null)
                handled = true
                continue
            }

            setHoveredSettingsTarget(target)
            handled = true

            if (event.kind !== 'press' || event.button !== 'left') {
                continue
            }

            if (target.kind === 'category') {
                setSettingsFocusArea('categories')
                setSelectedSettingsCategoryIndex(target.index)
                setSelectedSettingsFieldIndex(0)
            } else if (target.kind === 'field') {
                const field = settingsCategories[selectedSettingsCategoryIndex]?.fields[target.index]?.field

                setSettingsFocusArea('fields')
                setSelectedSettingsFieldIndex(target.index)

                if (field) {
                    openSettingsFieldEditor(field)
                }
            } else {
                setSettingsFocusArea('buttons')
                setSelectedSettingsButtonIndex(getSettingsButtonIndex(target.button))
                selectSettingsButton(target.button)
            }
        }

        return handled
    }, [absorbModalMouseWheelValue, openSettingsFieldEditor, selectSettingsButton, selectedSettingsCategoryIndex, settingsLayout, stage])

    const handleSetupWizardMouseValue = useCallback((value: string): boolean => {
        if (stage !== 'setup-wizard') {
            return false
        }

        if (absorbModalMouseWheelValue(value)) {
            return true
        }

        const events = parseMouseEvents(value)
        let handled = false

        for (const event of events) {
            if (event.kind === 'wheel') {
                continue
            }

            const target = getSetupWizardMouseTarget(event, setupWizardLayout)

            if (!target) {
                setHoveredSetupWizardTarget(null)
                handled = true
                continue
            }

            setHoveredSetupWizardTarget(target)
            handled = true

            if (!shouldActivateSetupWizardMouseEvent(event)) {
                continue
            }

            if (target.kind === 'field') {
                const field = currentSetupWizardFields[target.index]

                setSetupWizardFocusArea('fields')
                setSelectedSetupWizardFieldIndex(target.index)

                if (field) {
                    openSettingsFieldEditor(field, 'setup-wizard')
                }
            } else {
                setSetupWizardFocusArea('buttons')
                setSetupWizardButtonFocus(target.button === 'previous' ? 'previous' : target.button === 'retest' ? 'retest' : target.button === 'skipRecommended' ? 'secondary' : 'primary')
                void selectSetupWizardButton(target.button)
            }
        }

        return handled
    }, [absorbModalMouseWheelValue, currentSetupWizardFields, openSettingsFieldEditor, selectSetupWizardButton, setupWizardLayout, stage])

    const handleSettingsEditMouseValue = useCallback((value: string): boolean => {
        if (stage !== 'settings-edit' || !settingsEdit) {
            return false
        }

        if (absorbModalMouseWheelValue(value)) {
            return true
        }

        const events = parseMouseEvents(value)
        let handled = false

        for (const event of events) {
            if (event.kind === 'wheel') {
                continue
            }

            setSettingsEditKeyboardTarget(null)
            const target = getSettingsEditMouseTarget(event, settingsEditLayout)

            if (!target) {
                setHoveredSettingsEditTarget(null)
                handled = true
                continue
            }

            setHoveredSettingsEditTarget(target)
            handled = true

            if (event.kind !== 'press' || event.button !== 'left') {
                continue
            }

            if (target.kind === 'option') {
                setSettingsEdit(current => current ? { ...current, selectedOptionIndex: target.index } : current)
                setSettingsEditError(null)
            } else {
                if (target.kind === 'confirm') {
                    applySettingsEdit()
                } else {
                    cancelSettingsEdit()
                }
            }
        }

        return handled
    }, [absorbModalMouseWheelValue, applySettingsEdit, cancelSettingsEdit, settingsEdit, settingsEditLayout, stage])

    const handleTaskSelectionMouseValue = useCallback((value: string): boolean => {
        if (stage !== 'select-task' && stage !== 'select-glossary') {
            return false
        }

        if (absorbModalMouseWheelValue(value)) {
            return true
        }

        const events = parseMouseEvents(value)
        let handled = false

        for (const event of events) {
            if (event.kind === 'wheel') {
                continue
            }

            setTaskSelectionKeyboardTarget(null)
            const target = getTaskSelectionMouseTarget(event, taskSelectionLayout)

            if (!target) {
                setHoveredTaskSelectionTarget(null)
                handled = true
                continue
            }

            setHoveredTaskSelectionTarget(target)
            handled = true

            if (target.kind === 'choice') {
                if (stage === 'select-glossary') {
                    setSelectedGlossaryIndex(target.index)
                } else {
                    setSelectedTaskIndex(target.index)
                }

                if (event.kind === 'press' && event.button === 'left') {
                    if (stage === 'select-glossary') {
                        startSelectedGlossary(target.index)
                    } else {
                        startSelectedTask(target.index)
                    }
                }
            } else if (target.kind === 'confirm') {
                if (event.kind === 'press' && event.button === 'left') {
                    if (stage === 'select-glossary') {
                        startSelectedGlossary(selectedGlossaryIndex)
                    } else {
                        startSelectedTask(selectedTaskIndex)
                    }
                }
            } else if (target.kind === 'cancel') {
                if (event.kind === 'press' && event.button === 'left') {
                    void returnToStartMenu()
                }
            } else if (event.kind === 'press' && event.button === 'left') {
                void changeToDirectorySelection()
            }
        }

        return handled
    }, [absorbModalMouseWheelValue, changeToDirectorySelection, returnToStartMenu, selectedGlossaryIndex, selectedTaskIndex, stage, startSelectedGlossary, startSelectedTask, taskSelectionLayout])

    const handleMouseWheelValue = useCallback((value: string) => {
        const now = Date.now()
        const events = parseMouseWheelEvents(value)

        if (events.length === 0) {
            return false
        }

        for (const [signature, timestamp] of recentMouseWheelEventsRef.current) {
            if (now - timestamp > 20) {
                recentMouseWheelEventsRef.current.delete(signature)
            }
        }

        let handled = false

        for (const event of events) {
            const signature = `${event.direction}:${event.x ?? '-'}:${event.y ?? '-'}`

            if (now - (recentMouseWheelEventsRef.current.get(signature) ?? 0) <= 20) {
                continue
            }

            recentMouseWheelEventsRef.current.set(signature, now)

            const target: PaneScrollTarget = showPane
                ? getPaneScrollTarget(event, dualPaneLayout, {
                    panes: showWorkerPanes ? workerPanes : [],
                    scrollViewportRows: paneViewportRows,
                    showReviewPane,
                    showWorkerPanes,
                    scrollOffsetFromTop: paneScrollOffsetFromTop,
                })
                : { kind: 'transcript' }

            switch (target.kind) {
                case 'worker-pane':
                    {
                        const delta = event.direction === 'up' ? -SCROLL_STEP_LINES : SCROLL_STEP_LINES
                        const currentOffset = workerPaneScrollOffsets[target.slotIndex] ?? 0
                        const lineCount = workerPaneContentRows.get(target.slotIndex) ?? 0

                        if (canScrollTopViewport(currentOffset, delta, lineCount, workerPaneViewportRows)) {
                            scrollWorkerPaneBy(target.slotIndex, delta)
                        } else {
                            scrollPaneBy(delta)
                        }
                    }
                    break
                case 'review-pane':
                    {
                        const delta = event.direction === 'up' ? -SCROLL_STEP_LINES : SCROLL_STEP_LINES

                        if (canScrollTopViewport(reviewPaneScrollOffset, delta, reviewPaneContentRows, reviewPaneViewportRows)) {
                            scrollReviewPaneBy(delta)
                        } else {
                            scrollPaneBy(delta)
                        }
                    }
                    break
                case 'left-pane':
                    scrollPaneBy(event.direction === 'up' ? -SCROLL_STEP_LINES : SCROLL_STEP_LINES)
                    break
                case 'transcript':
                    scrollBy(event.direction === 'up' ? SCROLL_STEP_LINES : -SCROLL_STEP_LINES)
                    break
            }

            handled = true
        }

        return handled
    }, [canScrollTopViewport, dualPaneLayout, paneScrollOffsetFromTop, paneViewportRows, reviewPaneContentRows, reviewPaneScrollOffset, reviewPaneViewportRows, scrollBy, scrollPaneBy, scrollReviewPaneBy, scrollWorkerPaneBy, showPane, showReviewPane, showWorkerPanes, workerPaneContentRows, workerPaneScrollOffsets, workerPaneViewportRows, workerPanes])

    useEffect(() => {
        setScrollOffsetFromBottom(current => clampScrollOffset(current, transcriptLines.length, transcriptViewportRows))
    }, [transcriptLines.length, transcriptViewportRows])

    useEffect(() => {
        setPaneScrollOffsetFromTop(current => clampTopScrollOffset(current, paneContentRows, paneViewportRows))
    }, [paneContentRows, paneViewportRows])

    useEffect(() => {
        setWorkerPaneScrollOffsets(current => {
            const next: Record<number, number> = {}

            for (const pane of workerPanes) {
                const contentRowsForPane = workerPaneContentRows.get(pane.slotIndex) ?? 0
                next[pane.slotIndex] = clampTopScrollOffset(current[pane.slotIndex] ?? 0, contentRowsForPane, workerPaneViewportRows)
            }

            return next
        })
    }, [workerPaneContentRows, workerPaneViewportRows, workerPanes])

    useEffect(() => {
        setReviewPaneScrollOffset(current => clampTopScrollOffset(current, reviewPaneContentRows, reviewPaneViewportRows))
    }, [reviewPaneContentRows, reviewPaneViewportRows])

    useEffect(() => {
        return () => {
            if (tokenPulseTimerRef.current) {
                clearTimeout(tokenPulseTimerRef.current)
            }

            if (exitPromptTimerRef.current) {
                clearTimeout(exitPromptTimerRef.current)
            }

            if (clipboardMessageTimerRef.current) {
                clearTimeout(clipboardMessageTimerRef.current)
            }
        }
    }, [])

    useInput((inputValue, key) => {
        if (absorbModalMouseWheelValue(inputValue)) {
            return
        }

        if (handleMouseWheelValue(inputValue)) {
            return
        }

        if (isMouseInput(inputValue)) {
            return
        }

        if (Date.now() - lastRawMouseWheelAtRef.current < 80 && (key.upArrow || key.downArrow || key.pageUp || key.pageDown)) {
            return
        }

        if (key.ctrl && inputValue === 'c') {
            requestExit()
            return
        }

        if (stage === 'settings-edit' && settingsEdit) {
            const definition = settingsFieldDefinitions.get(settingsEdit.field)
            const options = getSettingsEditOptions(definition?.kind ?? 'string')

            if (key.escape) {
                handleEscape()
                return
            }

            if (key.tab || inputValue === '\t') {
                setHoveredSettingsEditTarget(null)
                setSettingsEditKeyboardTarget(current => getNextSettingsEditKeyboardTarget(current, settingsEdit.selectedOptionIndex, options.length))
                setSettingsEditError(null)
                return
            }

            if ((key.leftArrow || key.rightArrow) && (displayedSettingsEditTarget?.kind === 'confirm' || displayedSettingsEditTarget?.kind === 'cancel')) {
                setHoveredSettingsEditTarget(null)
                setSettingsEditKeyboardTarget(getNextSettingsEditButtonTarget(displayedSettingsEditTarget, key.leftArrow ? -1 : 1))
                return
            }

            if (options.length > 0) {
                const delta = getTaskSelectionDelta(inputValue, key)

                if (delta !== 0) {
                    setSettingsEditKeyboardTarget(null)
                    setHoveredSettingsEditTarget(null)
                    setSettingsEdit(current => current
                        ? { ...current, selectedOptionIndex: (current.selectedOptionIndex + options.length + delta) % options.length }
                        : current)
                    setSettingsEditError(null)
                    return
                }
            }

            if (key.return) {
                if (displayedSettingsEditTarget?.kind === 'cancel') {
                    cancelSettingsEdit()
                } else {
                    applySettingsEdit()
                }
                return
            }

            if (isPasteInput(inputValue, key) && options.length === 0 && !settingsEditKeyboardTarget) {
                void pasteIntoSettingsEdit()
                return
            }

            if ((key.backspace || key.delete) && options.length === 0 && !settingsEditKeyboardTarget) {
                setHoveredSettingsEditTarget(null)
                setSettingsEdit(current => current ? { ...current, input: current.input.slice(0, -1) } : current)
                setSettingsEditError(null)
                return
            }

            if (inputValue && !key.ctrl && !key.meta && options.length === 0 && !settingsEditKeyboardTarget) {
                setHoveredSettingsEditTarget(null)
                setSettingsEdit(current => current ? { ...current, input: `${current.input}${inputValue}` } : current)
                setSettingsEditError(null)
            }

            return
        }

        if (key.escape) {
            handleEscape()
            return
        }

        if (stage === 'start-menu') {
            const delta = getTaskSelectionDelta(inputValue, key)

            if (delta !== 0 || inputValue === '\t') {
                setHoveredStartMenuChoice(null)
                setSelectedStartMenuIndex(current => (current + START_MENU_CHOICES.length + (delta === 0 ? 1 : delta)) % START_MENU_CHOICES.length)
                return
            }

            if (key.return) {
                startMenuChoice(getStartMenuChoiceByIndex(selectedStartMenuIndex))
                return
            }

            return
        }

        if (stage === 'settings') {
            if (key.tab || inputValue === '\t') {
                setHoveredSettingsTarget(null)
                setSettingsFocusArea(current => current === 'categories' ? 'fields' : current === 'fields' ? 'buttons' : 'categories')
                return
            }

            if (key.leftArrow || key.rightArrow) {
                setHoveredSettingsTarget(null)
                if (settingsFocusArea === 'buttons') {
                    setSelectedSettingsButtonIndex(current => (current + 2 + (key.leftArrow ? -1 : 1)) % 2)
                } else {
                    setSettingsFocusArea(getHorizontalSettingsFocusArea(settingsFocusArea, key.leftArrow ? -1 : 1))
                }
                return
            }

            const delta = getTaskSelectionDelta(inputValue, key)

            if (delta !== 0) {
                setHoveredSettingsTarget(null)

                if (settingsFocusArea === 'categories') {
                    setSelectedSettingsCategoryIndex(current => {
                        const next = (current + settingsCategories.length + delta) % settingsCategories.length
                        setSelectedSettingsFieldIndex(0)
                        return next
                    })
                } else if (settingsFocusArea === 'fields') {
                    const fieldCount = Math.max(1, currentSettingsFields.length)
                    setSelectedSettingsFieldIndex(current => (current + fieldCount + delta) % fieldCount)
                } else {
                    setSelectedSettingsButtonIndex(current => (current + 2 + delta) % 2)
                }

                return
            }

            if (key.return) {
                if (settingsFocusArea === 'fields' && selectedSettingsField) {
                    openSettingsFieldEditor(selectedSettingsField.field)
                } else if (settingsFocusArea === 'buttons') {
                    selectSettingsButton(getSettingsButtonByIndex(selectedSettingsButtonIndex))
                }

                return
            }

            return
        }

        if (stage === 'setup-wizard') {
            if (key.tab || inputValue === '\t') {
                setHoveredSetupWizardTarget(null)
                setSetupWizardFocusArea(current => current === 'fields' ? 'buttons' : currentSetupWizardFields.length > 0 ? 'fields' : 'buttons')
                return
            }

            if (key.leftArrow || key.rightArrow) {
                setHoveredSetupWizardTarget(null)

                if (setupWizardFocusArea === 'buttons') {
                    setSetupWizardButtonFocus(current => getNextSetupWizardButtonFocus(current, selectedSetupWizardStepIndex, key.leftArrow ? -1 : 1))
                    return
                }

                if (key.rightArrow) {
                    void selectSetupWizardButton(getSetupWizardPrimaryButton(selectedSetupWizardStepIndex))
                } else if (layoutPreviousButtonEnabled(selectedSetupWizardStepIndex)) {
                    void selectSetupWizardButton('previous')
                }
                return
            }

            const delta = getTaskSelectionDelta(inputValue, key)

            if (delta !== 0) {
                setHoveredSetupWizardTarget(null)

                if (setupWizardFocusArea === 'fields' && currentSetupWizardFields.length > 0) {
                    setSelectedSetupWizardFieldIndex(current => (current + currentSetupWizardFields.length + delta) % currentSetupWizardFields.length)
                }

                return
            }

            if (key.return) {
                if (setupWizardFocusArea === 'fields' && selectedSetupWizardField) {
                    openSettingsFieldEditor(selectedSetupWizardField, 'setup-wizard')
                } else {
                    void selectSetupWizardButton(getFocusedSetupWizardButton(setupWizardButtonFocus, selectedSetupWizardStepIndex))
                }

                return
            }

            return
        }

        if (key.home) {
            setScrollOffsetFromBottom(maxScrollOffset)
            return
        }

        if (key.end) {
            setScrollOffsetFromBottom(0)
            return
        }

        if (stage === 'select-task') {
            const visibleTasks = visibleTaskChoices
            const choiceCount = getTaskSelectionChoiceCount(visibleTasks)
            const taskSelectionDelta = getTaskSelectionDelta(inputValue, key)

            if (key.tab || inputValue === '\t') {
                setHoveredTaskSelectionTarget(null)
                setTaskSelectionKeyboardTarget(current => getNextTaskSelectionKeyboardTarget(current, selectedTaskIndex))
                return
            }

            if ((key.leftArrow || key.rightArrow) && isTaskSelectionButtonTarget(displayedTaskSelectionTarget)) {
                setHoveredTaskSelectionTarget(null)
                setTaskSelectionKeyboardTarget(getNextTaskSelectionButtonTarget(displayedTaskSelectionTarget, key.leftArrow ? -1 : 1))
                return
            }

            if (taskSelectionDelta !== 0) {
                setTaskSelectionKeyboardTarget(null)
                setHoveredTaskSelectionTarget(null)
                setSelectedTaskIndex(current => (current + choiceCount + taskSelectionDelta) % choiceCount)
                return
            }

            if (key.return) {
                if (displayedTaskSelectionTarget?.kind === 'cancel') {
                    void returnToStartMenu()
                } else if (displayedTaskSelectionTarget?.kind === 'change-directory') {
                    void changeToDirectorySelection()
                } else {
                    startSelectedTask(displayedTaskSelectionTarget?.kind === 'choice' ? displayedTaskSelectionTarget.index : selectedTaskIndex)
                }
                return
            }

            if (inputValue === 'd' || inputValue === '/dir') {
                void changeToDirectorySelection()
                return
            }

            const keyboardScrollDelta = getKeyboardScrollDelta(inputValue, key, transcriptViewportRows)
            if (keyboardScrollDelta !== 0) {
                scrollBy(keyboardScrollDelta)
            }

            return
        }

        if (stage === 'select-glossary') {
            const visibleTasks = visibleGlossaryChoices
            const choiceCount = getTaskSelectionChoiceCount(visibleTasks)
            const taskSelectionDelta = getTaskSelectionDelta(inputValue, key)

            if (key.tab || inputValue === '\t') {
                setHoveredTaskSelectionTarget(null)
                setTaskSelectionKeyboardTarget(current => getNextTaskSelectionKeyboardTarget(current, selectedGlossaryIndex))
                return
            }

            if ((key.leftArrow || key.rightArrow) && isTaskSelectionButtonTarget(displayedTaskSelectionTarget)) {
                setHoveredTaskSelectionTarget(null)
                setTaskSelectionKeyboardTarget(getNextTaskSelectionButtonTarget(displayedTaskSelectionTarget, key.leftArrow ? -1 : 1))
                return
            }

            if (taskSelectionDelta !== 0) {
                setTaskSelectionKeyboardTarget(null)
                setHoveredTaskSelectionTarget(null)
                setSelectedGlossaryIndex(current => (current + choiceCount + taskSelectionDelta) % choiceCount)
                return
            }

            if (key.return) {
                if (displayedTaskSelectionTarget?.kind === 'cancel') {
                    void returnToStartMenu()
                } else if (displayedTaskSelectionTarget?.kind === 'change-directory') {
                    void changeToDirectorySelection()
                } else {
                    startSelectedGlossary(displayedTaskSelectionTarget?.kind === 'choice' ? displayedTaskSelectionTarget.index : selectedGlossaryIndex)
                }
                return
            }

            if (inputValue === 'd' || inputValue === '/dir') {
                void changeToDirectorySelection()
                return
            }

            const keyboardScrollDelta = getKeyboardScrollDelta(inputValue, key, transcriptViewportRows)
            if (keyboardScrollDelta !== 0) {
                scrollBy(keyboardScrollDelta)
            }

            return
        }

        if (stage === 'confirm-config') {
            if (key.tab || inputValue === '\t' || key.leftArrow || key.rightArrow) {
                setHoveredConfigMismatchChoice(null)
                setConfigMismatchKeyboardChoice(displayedConfigMismatchChoice === 'back' ? 'use-task-config' : 'back')
                return
            }

            const choice = getConfigMismatchChoice(inputValue, key)

            if (choice) {
                selectConfigMismatchChoice(key.return ? displayedConfigMismatchChoice ?? choice : choice)
                return
            }

            return
        }

        if (stage === 'task-complete') {
            if (key.return) {
                void returnToStartMenu()
            }

            return
        }

        const keyboardScrollDelta = getKeyboardScrollDelta(inputValue, key, transcriptViewportRows)
        if (keyboardScrollDelta !== 0) {
            scrollBy(keyboardScrollDelta)
            return
        }

        if (stage === 'checking-setup' || stage === 'initializing' || stage === 'preflighting' || stage === 'exporting' || stage === 'running') {
            return
        }

        if (stage === 'extracting') {
            return
        }

        if (key.return) {
            if (stage === 'select-directory') {
                void selectDirectory(input)
            } else {
                // Currently dead code: Ask mode is disabled; kept for possible future reuse.
                void submitPrompt(input)
            }
            return
        }

        if (isPasteInput(inputValue, key)) {
            void pasteIntoPromptInput()
            return
        }

        if (key.backspace || key.delete) {
            setInput(current => current.slice(0, -1))
            return
        }

        if (inputValue && !key.ctrl && !key.meta) {
            setInput(current => `${current}${inputValue}`)
        }
    })

    useEffect(() => {
        const writeEnable = () => {
            stdout.write(TERMINAL_MOUSE_ENABLE_SEQUENCE)
        }

        const timers = MOUSE_MODE_ENABLE_DELAYS_MS.map(delayMs => setTimeout(() => {
            writeEnable()
        }, delayMs))

        return () => {
            for (const timer of timers) {
                clearTimeout(timer)
            }

            stdout.write(TERMINAL_MOUSE_DISABLE_SEQUENCE)
        }
    }, [showPane, stdout])

    useEffect(() => {
        const handleData = (data: Buffer|string) => {
            const value = data.toString('utf8')

            if (handleStartMenuMouseValue(value)) {
                return
            }

            if (handleSettingsEditMouseValue(value)) {
                return
            }

            if (handleSettingsMouseValue(value)) {
                return
            }

            if (handleSetupWizardMouseValue(value)) {
                return
            }

            if (handleTaskSelectionMouseValue(value)) {
                return
            }

            if (handleConfigMismatchMouseValue(value)) {
                return
            }

            if (handleTaskCompleteMouseValue(value)) {
                return
            }

            if (handleMouseWheelValue(value)) {
                lastRawMouseWheelAtRef.current = Date.now()
            }
        }

        stdin.on('data', handleData)

        return () => {
            stdin.off('data', handleData)
        }
    }, [handleConfigMismatchMouseValue, handleMouseWheelValue, handleSettingsEditMouseValue, handleSettingsMouseValue, handleSetupWizardMouseValue, handleStartMenuMouseValue, handleTaskCompleteMouseValue, handleTaskSelectionMouseValue, stdin])

    return (
        <Box flexDirection="column" height={rows} paddingX={1} width="100%">
            <Text color="cyan" bold wrap="truncate-end">
                {stage === 'checking-setup' || stage === 'start-menu' || stage === 'settings' || stage === 'settings-edit' || stage === 'setup-wizard'
                    ? text.appTitle
                    : mode === 'translation'
                        ? text.translationAgentTitle
                        : text.glossaryAgentTitle}
            </Text>
            <Text dimColor wrap="truncate-end">
                {text.scope}: {runDirectory}
            </Text>
            {stage === 'start-menu' ? (
                <StartMenu
                    height={contentRows}
                    hoveredChoice={hoveredStartMenuChoice}
                    selectedIndex={selectedStartMenuIndex}
                    text={text}
                    terminalColumns={columns}
                />
            ) : stage === 'setup-wizard' ? (
                <SetupWizardPage
                    apiTest={setupWizardApiTest}
                    config={settingsConfig}
                    error={settingsLoadError}
                    fieldIndex={selectedSetupWizardFieldIndex}
                    focusArea={setupWizardFocusArea}
                    height={contentRows}
                    hoveredTarget={hoveredSetupWizardTarget}
                    layout={setupWizardLayout}
                    preferences={settingsPreferences}
                    saveMessage={settingsSaveMessage}
                    selectedButton={setupWizardButtonFocus}
                    stepIndex={selectedSetupWizardStepIndex}
                    text={text}
                />
            ) : stage === 'settings' || stage === 'settings-edit' ? (
                <Box height={contentRows} width="100%">
                    {stage === 'settings-edit' && settingsEdit ? (
                        <SettingsEditDialog
                            config={settingsConfig}
                            edit={settingsEdit}
                            error={settingsEditError}
                            height={contentRows}
                            hoveredTarget={displayedSettingsEditTarget}
                            layout={settingsEditLayout}
                            preferences={settingsPreferences}
                            text={text}
                            terminalColumns={columns}
                        />
                    ) : (
                        <SettingsPage
                            config={settingsConfig}
                            error={settingsLoadError}
                            focusArea={settingsFocusArea}
                            height={contentRows}
                            hoveredTarget={hoveredSettingsTarget}
                            layout={settingsLayout}
                            saveMessage={settingsSaveMessage}
                            selectedButtonIndex={selectedSettingsButtonIndex}
                            selectedCategoryIndex={selectedSettingsCategoryIndex}
                            selectedFieldIndex={selectedSettingsFieldIndex}
                            preferences={settingsPreferences}
                            text={text}
                        />
                    )}
                </Box>
            ) : stage === 'select-task' ? (
                <TaskSelectionDialog
                    hoveredTarget={displayedTaskSelectionTarget}
                    height={contentRows}
                    mode={mode}
                    runDirectory={runDirectory}
                    selectedIndex={selectedTaskIndex}
                    tasks={taskChoices}
                    text={text}
                    terminalColumns={columns}
                />
            ) : stage === 'select-glossary' ? (
                <TaskSelectionDialog
                    hoveredTarget={displayedTaskSelectionTarget}
                    height={contentRows}
                    mode={mode}
                    newTaskLabel={text.taskDialog.noGlossary}
                    promptText={visibleGlossaryChoices.length === 0 ? text.taskDialog.noMatchingGlossary : text.taskDialog.selectGlossaryPrompt}
                    runDirectory={runDirectory}
                    selectedIndex={selectedGlossaryIndex}
                    text={text}
                    title={text.taskDialog.selectGlossaryTitle}
                    tasks={glossaryChoices}
                    terminalColumns={columns}
                />
            ) : stage === 'confirm-config' && pendingConfigResume ? (
                <ConfigMismatchDialog
                    differences={pendingConfigResume.differences}
                    height={contentRows}
                    hoveredChoice={displayedConfigMismatchChoice}
                    layout={configMismatchLayout}
                    task={pendingConfigResume.task}
                    text={text}
                    terminalColumns={columns}
                />
            ) : stage === 'task-complete' ? (
                <TaskCompleteDialog
                    height={contentRows}
                    hoveredTarget={hoveredTaskCompleteTarget}
                    layout={taskCompleteLayout}
                    mode={mode}
                    text={text}
                />
            ) : showPane ? (
                <Box flexDirection="row" height={contentRows} overflowY="hidden">
                    {showReviewPane ? (
                        <ScrollableReviewPane
                            height={contentRows}
                            pane={reviewPane}
                            paneScrollOffset={reviewPaneScrollOffset}
                            text={text}
                            width={dualPaneLayout.paneWidth}
                        />
                    ) : (
                        <ScrollableWorkerPaneList
                            height={contentRows}
                            panes={workerPanes}
                            scrollOffset={paneScrollOffsetFromTop}
                            text={text}
                            workerPaneScrollOffsets={workerPaneScrollOffsets}
                            title={mode === 'translation' ? text.pane.translationWorkerAgents : text.pane.workerAgents}
                            width={dualPaneLayout.paneWidth}
                        />
                    )}
                    <TranscriptPane
                        height={contentRows}
                        lines={transcriptLines}
                        marginLeft={LAYOUT_GAP_COLUMNS}
                        maxScrollOffset={maxScrollOffset}
                        scrollOffsetFromBottom={scrollOffsetFromBottom}
                        width={dualPaneLayout.transcriptWidth}
                    />
                </Box>
            ) : (
                <TranscriptPane
                    height={transcriptViewportRows}
                    lines={transcriptLines}
                    maxScrollOffset={maxScrollOffset}
                    scrollOffsetFromBottom={scrollOffsetFromBottom}
                    width={Math.max(1, columns - 2)}
                />
            )}
            <PromptLine stage={stage} input={input} mode={mode} text={text} />
            <Text color={exitPromptVisible || clipboardMessage ? 'yellow' : undefined} dimColor={!exitPromptVisible && !clipboardMessage} wrap="truncate-end">
                {exitPromptVisible ? text.message.ctrlCAgain : clipboardMessage ?? getHelpText(stage, text)}
            </Text>
            <StatusLine
                apiStatus={apiStatus}
                columns={columns}
                maxScrollOffset={maxScrollOffset}
                scrollOffsetFromBottom={scrollOffsetFromBottom}
                stage={stage}
                text={text}
                tokenPulse={tokenPulse}
                tokenUsage={tokenUsage}
            />
        </Box>
    )
}

function formatGlossaryPlan (plan: GlossaryPlan, text: TuiText): string {
    return [
        text.report.glossaryPreflightPlan,
        `${text.report.planId}: ${plan.plan_id}`,
        `${text.report.created}: ${plan.created_at}`,
        '',
        `${text.report.projectContext}:`,
        plan.shared_prompt_context.project_context,
        '',
        `${text.report.domainHint}: ${plan.shared_prompt_context.domain_hint}`,
        `${text.report.sourceTarget}: ${plan.shared_prompt_context.source_language} -> ${plan.shared_prompt_context.target_language}`,
        '',
        `${text.report.extractionPolicy}:`,
        `${text.report.termTypes}: ${plan.term_extraction_policy.term_types.join(', ')}`,
        `${text.report.entryTypes}: ${plan.term_extraction_policy.entry_types.join(', ')}`,
        `${text.report.defaults}: ${text.report.term}=${plan.term_extraction_policy.default_term_status}, ${text.report.entry}=${plan.term_extraction_policy.default_entry_status}`,
        `${text.report.evidenceRequired}: ${plan.term_extraction_policy.require_evidence ? text.report.yes : text.report.no}`,
        `${text.report.oneEntryOneClaim}: ${plan.term_extraction_policy.one_entry_one_claim ? text.report.yes : text.report.no}`,
        '',
        `${text.report.focus}:`,
        formatList(plan.shared_prompt_context.focus),
        '',
        `${text.report.cautions}:`,
        formatList(plan.shared_prompt_context.cautions),
        '',
        `${text.report.notes}:`,
        formatList(plan.shared_prompt_context.notes),
    ].join('\n')
}

function formatTranslationPreflight (preflight: TranslationPreflight, text: TuiText): string {
    return [
        text.report.translationPreflight,
        `${text.report.submitted}: ${preflight.submitted_at}`,
        `${text.report.domainHint}: ${preflight.domain_hint}`,
        `${text.report.sourceTarget}: ${preflight.source_language} -> ${preflight.target_language}`,
        '',
        `${text.report.textProfile}:`,
        formatList(preflight.text_profile),
        '',
        `${text.report.styleGuidance}:`,
        formatList(preflight.style_guidance),
        '',
        `${text.report.formatProtection}:`,
        formatList(preflight.format_protection),
        '',
        `${text.report.glossaryUsage}:`,
        formatList(preflight.glossary_usage),
        '',
        `${text.report.contextUsage}:`,
        formatList(preflight.context_usage),
        '',
        `${text.report.qualityCautions}:`,
        formatList(preflight.quality_cautions),
    ].join('\n')
}

function formatTranslationExportResult (result: TranslationExportResult, text: TuiText): string {
    return [
        `${text.report.translationExportWritten}: ${result.outputPath}`,
        `${text.report.translatedValues}: ${result.translatedCount}/${result.totalKeys}`,
        `${text.report.preservedOriginalValues}: ${result.preservedCount}`,
    ].join('\n')
}

function formatList (items: string[]): string {
    return items.map((item, index) => `${index + 1}. ${item}`).join('\n')
}

export function flattenTranscriptEntries (entries: TranscriptEntry[], text: TuiText = getTuiText()): DisplayLine[] {
    return entries.flatMap(entry => {
        const { color, label } = getTranscriptStyle(entry, text)
        const textLines = entry.text.split('\n')
        const lines: DisplayLine[] = [
            {
                id: `${entry.id}-label`,
                text: label,
                color,
                bold: true,
            },
            ...textLines.map((line, index) => ({
                id: `${entry.id}-text-${index}`,
                text: line,
                color: entry.kind === 'error' ? 'red' : undefined,
            })),
            {
                id: `${entry.id}-spacer`,
                text: '',
                dimColor: true,
            },
        ]

        return lines
    })
}

export function getVisibleDisplayLines (
    lines: DisplayLine[],
    viewportHeight: number,
    scrollOffsetFromBottom: number,
): DisplayLine[] {
    const safeViewportHeight = Math.max(1, viewportHeight)
    const safeOffset = clampScrollOffset(scrollOffsetFromBottom, lines.length, safeViewportHeight)
    const end = Math.max(0, lines.length - safeOffset)
    const start = Math.max(0, end - safeViewportHeight)

    return lines.slice(start, end)
}

export function getVisibleDisplayLinesFromTop (
    lines: DisplayLine[],
    viewportHeight: number,
    scrollOffsetFromTop: number,
): DisplayLine[] {
    const safeViewportHeight = Math.max(1, viewportHeight)
    const safeOffset = clampTopScrollOffset(scrollOffsetFromTop, lines.length, safeViewportHeight)

    return lines.slice(safeOffset, safeOffset + safeViewportHeight)
}

export function getMaxScrollOffset (lineCount: number, viewportHeight: number): number {
    return Math.max(0, lineCount - Math.max(1, viewportHeight))
}

export function clampScrollOffset (offset: number, lineCount: number, viewportHeight: number): number {
    return Math.max(0, Math.min(offset, getMaxScrollOffset(lineCount, viewportHeight)))
}

export function getMaxTopScrollOffset (lineCount: number, viewportHeight: number): number {
    return Math.max(0, lineCount - Math.max(1, viewportHeight))
}

export function clampTopScrollOffset (offset: number, lineCount: number, viewportHeight: number): number {
    return Math.max(0, Math.min(offset, getMaxTopScrollOffset(lineCount, viewportHeight)))
}

export function calculateDualPaneLayout (columns: number, contentRows: number): DualPaneLayout {
    const contentWidth = getRootContentWidth(columns)
    const gap = LAYOUT_GAP_COLUMNS
    const transcriptWidth = contentWidth >= 90
        ? Math.max(30, Math.floor(contentWidth * 0.30))
        : Math.max(18, Math.floor(contentWidth * 0.34))
    const safeTranscriptWidth = Math.min(transcriptWidth, Math.max(1, contentWidth - gap - 1))
    const paneWidth = Math.max(1, contentWidth - gap - safeTranscriptWidth)
    const leftStartColumn = ROOT_PADDING_COLUMNS + 1
    const leftEndColumn = leftStartColumn + paneWidth - 1
    const rightStartColumn = leftEndColumn + gap + 1
    const rightEndColumn = rightStartColumn + safeTranscriptWidth - 1

    return {
        contentStartRow: HEADER_ROWS + 1,
        contentEndRow: HEADER_ROWS + Math.max(1, contentRows),
        leftStartColumn,
        leftEndColumn,
        rightStartColumn,
        rightEndColumn,
        paneWidth,
        transcriptWidth: safeTranscriptWidth,
    }
}

export function calculateTaskSelectionDialogLayout ({
    height,
    taskCount,
    text = getTuiText(),
    terminalColumns,
}: {
    height: number
    taskCount: number
    text?: TuiText
    terminalColumns: number
}): TaskSelectionDialogLayout {
    const visibleTaskCount = Math.min(Math.max(0, taskCount), MAX_VISIBLE_TASK_CHOICES)
    const choiceCount = visibleTaskCount + 1
    const rootContentWidth = getRootContentWidth(terminalColumns)
    const dialogWidth = getTaskSelectionDialogWidth(terminalColumns)
    const dialogOuterRows = getTaskSelectionDialogOuterRows(visibleTaskCount)
    const startColumn = ROOT_PADDING_COLUMNS + 1 + getCenteredOffset(rootContentWidth, dialogWidth)
    const startRow = HEADER_ROWS + 1 + getCenteredOffset(height, dialogOuterRows)
    const buttonStartColumn = startColumn + 2
    const [confirmButton, cancelButton, changeDirectoryButton] = getButtonRowBounds(buttonStartColumn, getDialogContentWidth(dialogWidth), [
        text.taskDialog.ok,
        text.taskDialog.cancel,
        text.taskDialog.changeDirectory,
    ])
    const optionStartRow = startRow
        + TASK_DIALOG_BORDER_ROWS / 2
        + TASK_DIALOG_VERTICAL_PADDING_ROWS / 2
        + TASK_DIALOG_HEADER_ROWS
        + TASK_DIALOG_OPTION_MARGIN_TOP_ROWS
    const buttonRow = optionStartRow + choiceCount + TASK_DIALOG_BUTTON_MARGIN_TOP_ROWS

    return {
        startColumn,
        endColumn: startColumn + dialogWidth - 1,
        startRow,
        endRow: startRow + dialogOuterRows - 1,
        optionStartRow,
        optionEndRow: optionStartRow + choiceCount - 1,
        buttonRow,
        confirmButtonStartColumn: confirmButton.startColumn,
        confirmButtonEndColumn: confirmButton.endColumn,
        cancelButtonStartColumn: cancelButton.startColumn,
        cancelButtonEndColumn: cancelButton.endColumn,
        changeDirectoryButtonStartColumn: changeDirectoryButton.startColumn,
        changeDirectoryButtonEndColumn: changeDirectoryButton.endColumn,
        choiceCount,
    }
}

export function getTaskSelectionMouseTarget (
    event: Pick<MouseInputEvent, 'x'|'y'>,
    layout: TaskSelectionDialogLayout,
): TaskSelectionMouseTarget|null {
    if (event.x < layout.startColumn || event.x > layout.endColumn || event.y < layout.startRow || event.y > layout.endRow) {
        return null
    }

    if (event.y >= layout.optionStartRow && event.y <= layout.optionEndRow) {
        const index = event.y - layout.optionStartRow

        return index >= 0 && index < layout.choiceCount
            ? { kind: 'choice', index }
            : null
    }

    if (event.y === layout.buttonRow) {
        if (event.x >= layout.confirmButtonStartColumn && event.x <= layout.confirmButtonEndColumn) {
            return { kind: 'confirm' }
        }

        if (event.x >= layout.cancelButtonStartColumn && event.x <= layout.cancelButtonEndColumn) {
            return { kind: 'cancel' }
        }

        if (event.x >= layout.changeDirectoryButtonStartColumn && event.x <= layout.changeDirectoryButtonEndColumn) {
            return { kind: 'change-directory' }
        }
    }

    return null
}

export function calculateStartMenuLayout ({
    height,
    text = getTuiText(),
    terminalColumns,
}: {
    height: number
    text?: TuiText
    terminalColumns: number
}): StartMenuLayout {
    const rootContentWidth = getRootContentWidth(terminalColumns)
    const dialogWidth = getStartMenuDialogWidth(terminalColumns)
    const dialogOuterRows = getStartMenuDialogOuterRows()
    const startColumn = ROOT_PADDING_COLUMNS + 1 + getCenteredOffset(rootContentWidth, dialogWidth)
    const startRow = HEADER_ROWS + 1 + getCenteredOffset(height, dialogOuterRows)
    const safeButtonWidth = Math.min(getStartMenuButtonWidth(text), dialogWidth)
    const buttonColumn = startColumn + Math.max(0, Math.floor((dialogWidth - safeButtonWidth) / 2))
    const choiceStartRow = getStartMenuChoiceStartRow(startRow)

    return {
        startColumn: buttonColumn,
        endColumn: buttonColumn + safeButtonWidth - 1,
        startRow,
        endRow: startRow + dialogOuterRows - 1,
        choiceRows: START_MENU_CHOICES.map((choice, index) => ({
            choice,
            row: choiceStartRow + index * getStartMenuChoiceRowSpan(),
        })),
    }
}

export function getStartMenuMouseTarget (
    event: Pick<MouseInputEvent, 'x'|'y'>,
    layout: StartMenuLayout,
): StartMenuMouseTarget|null {
    if (event.x < layout.startColumn || event.x > layout.endColumn || event.y < layout.startRow || event.y > layout.endRow) {
        return null
    }

    const choiceRow = layout.choiceRows.find(row => row.row === event.y)

    return choiceRow ? { kind: 'choice', choice: choiceRow.choice } : null
}

function getStartMenuChoiceIndex (choice: StartMenuChoice): number {
    const index = START_MENU_CHOICES.findIndex(item => item === choice)

    return index === -1 ? 0 : index
}

function getStartMenuChoiceByIndex (index: number): StartMenuChoice {
    const normalizedIndex = (index + START_MENU_CHOICES.length) % START_MENU_CHOICES.length

    return START_MENU_CHOICES[normalizedIndex] ?? 'glossary'
}

export function calculateConfigMismatchLayout ({
    differenceCount,
    height,
    text = getTuiText(),
    terminalColumns,
}: {
    differenceCount: number
    height: number
    text?: TuiText
    terminalColumns: number
}): ConfigMismatchLayout {
    const rootContentWidth = getRootContentWidth(terminalColumns)
    const dialogWidth = getTaskSelectionDialogWidth(terminalColumns)
    const differenceRows = getConfigMismatchDifferenceRows(differenceCount, height)
    const outerRows = getConfigMismatchDialogOuterRows(differenceRows)
    const startColumn = ROOT_PADDING_COLUMNS + 1 + getCenteredOffset(rootContentWidth, dialogWidth)
    const startRow = HEADER_ROWS + 1 + getCenteredOffset(height, outerRows)
    const buttonStartColumn = startColumn + 2
    const [useTaskConfigButton, backButton] = getButtonRowBounds(buttonStartColumn, getDialogContentWidth(dialogWidth), [
        text.configMismatch.useTaskConfig,
        text.configMismatch.back,
    ])
    const differenceStartRow = getConfigMismatchDifferenceStartRow(startRow)
    const buttonRow = differenceStartRow + differenceRows + CONFIG_MISMATCH_BUTTON_MARGIN_TOP_ROWS

    return {
        startColumn,
        endColumn: startColumn + dialogWidth - 1,
        startRow,
        endRow: startRow + outerRows - 1,
        buttonRow,
        useTaskConfigButtonStartColumn: useTaskConfigButton.startColumn,
        useTaskConfigButtonEndColumn: useTaskConfigButton.endColumn,
        backButtonStartColumn: backButton.startColumn,
        backButtonEndColumn: backButton.endColumn,
    }
}

export function getConfigMismatchMouseTarget (
    event: Pick<MouseInputEvent, 'x'|'y'>,
    layout: ConfigMismatchLayout,
): ConfigMismatchMouseTarget|null {
    if (event.x < layout.startColumn || event.x > layout.endColumn || event.y < layout.startRow || event.y > layout.endRow) {
        return null
    }

    if (event.y !== layout.buttonRow) {
        return null
    }

    if (event.x >= layout.useTaskConfigButtonStartColumn && event.x <= layout.useTaskConfigButtonEndColumn) {
        return { kind: 'button', choice: 'use-task-config' }
    }

    if (event.x >= layout.backButtonStartColumn && event.x <= layout.backButtonEndColumn) {
        return { kind: 'button', choice: 'back' }
    }

    return null
}

export function calculateTaskCompleteLayout ({
    height,
    text = getTuiText(),
    terminalColumns,
}: {
    height: number
    text?: TuiText
    terminalColumns: number
}): TaskCompleteLayout {
    const rootContentWidth = getRootContentWidth(terminalColumns)
    const dialogWidth = getTaskSelectionDialogWidth(terminalColumns)
    const outerRows = getTaskCompleteDialogOuterRows()
    const startColumn = ROOT_PADDING_COLUMNS + 1 + getCenteredOffset(rootContentWidth, dialogWidth)
    const startRow = HEADER_ROWS + 1 + getCenteredOffset(height, outerRows)
    const buttonStartColumn = startColumn + 2
    const [backButton] = getButtonRowBounds(buttonStartColumn, getDialogContentWidth(dialogWidth), [
        text.taskComplete.backToMenu,
    ])
    const buttonRow = startRow
        + TASK_COMPLETE_BORDER_ROWS / 2
        + TASK_COMPLETE_VERTICAL_PADDING_ROWS / 2
        + TASK_COMPLETE_HEADER_ROWS
        + TASK_COMPLETE_BUTTON_MARGIN_TOP_ROWS

    return {
        startColumn,
        endColumn: startColumn + dialogWidth - 1,
        startRow,
        endRow: startRow + outerRows - 1,
        buttonRow,
        buttonStartColumn: backButton?.startColumn ?? buttonStartColumn,
        buttonEndColumn: backButton?.endColumn ?? buttonStartColumn,
    }
}

export function getTaskCompleteMouseTarget (
    event: Pick<MouseInputEvent, 'x'|'y'>,
    layout: TaskCompleteLayout,
): TaskCompleteMouseTarget|null {
    if (event.x < layout.startColumn || event.x > layout.endColumn || event.y < layout.startRow || event.y > layout.endRow) {
        return null
    }

    if (event.y === layout.buttonRow && event.x >= layout.buttonStartColumn && event.x <= layout.buttonEndColumn) {
        return { kind: 'button' }
    }

    return null
}

export function calculateSettingsLayout ({
    categoryCount,
    fieldCount,
    fieldHitCount = fieldCount,
    height,
    text = getTuiText(),
    terminalColumns,
}: {
    categoryCount: number
    fieldCount: number
    fieldHitCount?: number
    height: number
    text?: TuiText
    terminalColumns: number
}): SettingsLayout {
    const rootContentWidth = getRootContentWidth(terminalColumns)
    const availableWidth = Math.max(1, rootContentWidth - SETTINGS_DIALOG_SIDE_MARGIN_COLUMNS * 2)
    const dialogWidth = availableWidth < SETTINGS_DIALOG_MIN_WIDTH
        ? availableWidth
        : Math.min(SETTINGS_DIALOG_MAX_WIDTH, availableWidth)
    const contentWidth = Math.max(1, dialogWidth - 4)
    const leftWidth = Math.min(SETTINGS_DIALOG_CATEGORY_WIDTH, Math.max(12, Math.floor(contentWidth * 0.32)))
    const startColumn = ROOT_PADDING_COLUMNS + 1 + getCenteredOffset(rootContentWidth, dialogWidth)
    const outerRows = Math.max(
        SETTINGS_DIALOG_MIN_ROWS,
        SETTINGS_DIALOG_BORDER_ROWS
            + SETTINGS_DIALOG_VERTICAL_PADDING_ROWS
            + SETTINGS_DIALOG_HEADER_ROWS
            + SETTINGS_DIALOG_LIST_MARGIN_TOP_ROWS
            + Math.max(categoryCount, fieldCount)
            + SETTINGS_DIALOG_BUTTON_MARGIN_TOP_ROWS
            + SETTINGS_DIALOG_BUTTON_ROWS
            + SETTINGS_DIALOG_FOOTER_ROWS,
    )
    const startRow = HEADER_ROWS + 1 + getCenteredOffset(height, outerRows)
    const contentStartColumn = startColumn + 2
    const categoryStartRow = startRow
        + SETTINGS_DIALOG_BORDER_ROWS / 2
        + SETTINGS_DIALOG_VERTICAL_PADDING_ROWS / 2
        + SETTINGS_DIALOG_HEADER_ROWS
        + SETTINGS_DIALOG_LIST_MARGIN_TOP_ROWS
    const visibleRows = Math.max(categoryCount, fieldCount)
    const buttonRow = categoryStartRow + visibleRows + SETTINGS_DIALOG_BUTTON_MARGIN_TOP_ROWS
    const buttonStartColumn = contentStartColumn
    const [saveButton, backButton] = getButtonRowBounds(buttonStartColumn, getDialogContentWidth(dialogWidth), [
        text.settings.save,
        text.settings.back,
    ])

    return {
        startColumn,
        endColumn: startColumn + dialogWidth - 1,
        startRow,
        endRow: startRow + outerRows - 1,
        leftStartColumn: contentStartColumn,
        leftEndColumn: contentStartColumn + leftWidth - 1,
        rightStartColumn: contentStartColumn + leftWidth + 2,
        rightEndColumn: startColumn + dialogWidth - 3,
        categoryStartRow,
        categoryEndRow: categoryStartRow + Math.max(0, categoryCount - 1),
        fieldStartRow: categoryStartRow,
        fieldEndRow: categoryStartRow + Math.max(0, fieldCount - 1),
        fieldHitEndRow: categoryStartRow + Math.max(0, Math.min(fieldCount, fieldHitCount) - 1),
        buttonRow,
        saveButtonStartColumn: saveButton.startColumn,
        saveButtonEndColumn: saveButton.endColumn,
        backButtonStartColumn: backButton.startColumn,
        backButtonEndColumn: backButton.endColumn,
    }
}

export function getSettingsMouseTarget (
    event: Pick<MouseInputEvent, 'x'|'y'>,
    layout: SettingsLayout,
): SettingsMouseTarget|null {
    if (event.x < layout.startColumn || event.x > layout.endColumn || event.y < layout.startRow || event.y > layout.endRow) {
        return null
    }

    if (event.x >= layout.leftStartColumn && event.x <= layout.leftEndColumn && event.y >= layout.categoryStartRow && event.y <= layout.categoryEndRow) {
        return { kind: 'category', index: event.y - layout.categoryStartRow }
    }

    if (event.x >= layout.rightStartColumn && event.x <= layout.rightEndColumn && event.y >= layout.fieldStartRow && event.y <= layout.fieldHitEndRow) {
        return { kind: 'field', index: event.y - layout.fieldStartRow }
    }

    if (event.y === layout.buttonRow) {
        if (event.x >= layout.saveButtonStartColumn && event.x <= layout.saveButtonEndColumn) {
            return { kind: 'button', button: 'save' }
        }

        if (event.x >= layout.backButtonStartColumn && event.x <= layout.backButtonEndColumn) {
            return { kind: 'button', button: 'back' }
        }
    }

    return null
}

export function calculateSetupWizardLayout ({
    fieldCount,
    height,
    selectedStepIndex,
    text = getTuiText(),
    terminalColumns,
}: {
    fieldCount: number
    height: number
    selectedStepIndex: number
    text?: TuiText
    terminalColumns: number
}): SetupWizardLayout {
    const rootContentWidth = getRootContentWidth(terminalColumns)
    const availableWidth = Math.max(1, rootContentWidth - SETUP_WIZARD_DIALOG_SIDE_MARGIN_COLUMNS * 2)
    const dialogWidth = availableWidth < SETUP_WIZARD_DIALOG_MIN_WIDTH
        ? availableWidth
        : Math.min(SETUP_WIZARD_DIALOG_MAX_WIDTH, availableWidth)
    const contentWidth = Math.max(1, dialogWidth - 4)
    const leftWidth = Math.min(SETUP_WIZARD_STEP_WIDTH, Math.max(14, Math.floor(contentWidth * 0.3)))
    const startColumn = ROOT_PADDING_COLUMNS + 1 + getCenteredOffset(rootContentWidth, dialogWidth)
    const contentStartColumn = startColumn + 2
    const step = SETUP_WIZARD_STEPS[selectedStepIndex] ?? SETUP_WIZARD_STEPS[0]!
    const visibleRows = getSetupWizardVisibleRows(text)
    const outerRows = Math.max(
        SETUP_WIZARD_DIALOG_MIN_ROWS,
        SETUP_WIZARD_BORDER_ROWS
            + SETUP_WIZARD_VERTICAL_PADDING_ROWS
            + SETUP_WIZARD_HEADER_ROWS
            + SETUP_WIZARD_LIST_MARGIN_TOP_ROWS
            + visibleRows
            + SETUP_WIZARD_BUTTON_MARGIN_TOP_ROWS
            + SETUP_WIZARD_BUTTON_ROWS
            + SETUP_WIZARD_FOOTER_ROWS,
    )
    const startRow = HEADER_ROWS + 1 + getCenteredOffset(height, outerRows)
    const stepStartRow = startRow
        + SETUP_WIZARD_BORDER_ROWS / 2
        + SETUP_WIZARD_VERTICAL_PADDING_ROWS / 2
        + SETUP_WIZARD_HEADER_ROWS
        + SETUP_WIZARD_LIST_MARGIN_TOP_ROWS
    const buttonRow = stepStartRow + visibleRows + SETUP_WIZARD_BUTTON_MARGIN_TOP_ROWS
    const primaryButton = getSetupWizardPrimaryButton(selectedStepIndex)
    const previousEnabled = layoutPreviousButtonEnabled(selectedStepIndex)
    const hasRetestButton = step.id === 'apiTest'
    const hasSkipRecommendedButton = step.id === 'recommended'
    const buttonLabels = [
        text.setupWizard.previous,
        ...(hasRetestButton ? [text.setupWizard.retest] : []),
        ...(hasSkipRecommendedButton ? [text.setupWizard.skipRecommended] : []),
        getSetupWizardPrimaryButtonLabel(primaryButton, text),
    ]
    const buttonWidths = getRightAlignedButtonRowWidths(getDialogContentWidth(dialogWidth), buttonLabels)
    const buttonRowWidth = buttonWidths.reduce((total, width) => total + width, 0)
        + TASK_DIALOG_BUTTON_GAP_COLUMNS * Math.max(0, buttonWidths.length - 1)
    const buttonStartColumn = startColumn + dialogWidth - 3 - buttonRowWidth + 1
    const previousButton = {
        startColumn: buttonStartColumn,
        endColumn: buttonStartColumn + buttonWidths[0]! - 1,
    }
    const retestButton = hasRetestButton
        ? {
            startColumn: previousButton.endColumn + TASK_DIALOG_BUTTON_GAP_COLUMNS + 1,
            endColumn: previousButton.endColumn + TASK_DIALOG_BUTTON_GAP_COLUMNS + buttonWidths[1]!,
        }
        : null
    const secondaryWidthIndex = hasRetestButton ? 2 : 1
    const secondaryButton = hasSkipRecommendedButton
        ? {
            startColumn: previousButton.endColumn + TASK_DIALOG_BUTTON_GAP_COLUMNS + 1,
            endColumn: previousButton.endColumn + TASK_DIALOG_BUTTON_GAP_COLUMNS + buttonWidths[secondaryWidthIndex]!,
        }
        : null
    const primaryWidthIndex = 1 + (hasRetestButton ? 1 : 0) + (hasSkipRecommendedButton ? 1 : 0)
    const primaryStartColumn = secondaryButton
        ? secondaryButton.endColumn + TASK_DIALOG_BUTTON_GAP_COLUMNS + 1
        : retestButton
            ? retestButton.endColumn + TASK_DIALOG_BUTTON_GAP_COLUMNS + 1
            : previousButton.endColumn + TASK_DIALOG_BUTTON_GAP_COLUMNS + 1
    const primaryButtonBounds = {
        startColumn: primaryStartColumn,
        endColumn: primaryStartColumn + buttonWidths[primaryWidthIndex]! - 1,
    }

    return {
        startColumn,
        endColumn: startColumn + dialogWidth - 1,
        startRow,
        endRow: startRow + outerRows - 1,
        leftStartColumn: contentStartColumn,
        leftEndColumn: contentStartColumn + leftWidth - 1,
        rightStartColumn: contentStartColumn + leftWidth + 2,
        rightEndColumn: startColumn + dialogWidth - 3,
        stepStartRow,
        stepEndRow: stepStartRow + SETUP_WIZARD_STEPS.length - 1,
        fieldStartRow: stepStartRow + SETUP_WIZARD_STEP_HEADER_ROWS,
        fieldEndRow: fieldCount > 0
            ? stepStartRow + SETUP_WIZARD_STEP_HEADER_ROWS + fieldCount - 1
            : stepStartRow + SETUP_WIZARD_STEP_HEADER_ROWS - 1,
        buttonRow,
        previousButtonStartColumn: previousButton.startColumn,
        previousButtonEndColumn: previousButton.endColumn,
        retestButtonStartColumn: retestButton?.startColumn ?? null,
        retestButtonEndColumn: retestButton?.endColumn ?? null,
        secondaryButtonStartColumn: secondaryButton?.startColumn ?? null,
        secondaryButtonEndColumn: secondaryButton?.endColumn ?? null,
        primaryButtonStartColumn: primaryButtonBounds.startColumn,
        primaryButtonEndColumn: primaryButtonBounds.endColumn,
        primaryButton,
        previousEnabled,
    }
}

export function getSetupWizardMouseTarget (
    event: Pick<MouseInputEvent, 'x'|'y'>,
    layout: SetupWizardLayout,
): SetupWizardMouseTarget|null {
    if (event.x < layout.startColumn || event.x > layout.endColumn || event.y < layout.startRow || event.y > layout.endRow) {
        return null
    }

    if (event.x >= layout.rightStartColumn && event.x <= layout.rightEndColumn && event.y >= layout.fieldStartRow && event.y <= layout.fieldEndRow) {
        return { kind: 'field', index: event.y - layout.fieldStartRow }
    }

    if (event.y === layout.buttonRow) {
        if (layout.previousEnabled && event.x >= layout.previousButtonStartColumn && event.x <= layout.previousButtonEndColumn) {
            return { kind: 'button', button: 'previous' }
        }

        if (
            layout.retestButtonStartColumn !== null
            && layout.retestButtonEndColumn !== null
            && event.x >= layout.retestButtonStartColumn
            && event.x <= layout.retestButtonEndColumn
        ) {
            return { kind: 'button', button: 'retest' }
        }

        if (
            layout.secondaryButtonStartColumn !== null
            && layout.secondaryButtonEndColumn !== null
            && event.x >= layout.secondaryButtonStartColumn
            && event.x <= layout.secondaryButtonEndColumn
        ) {
            return { kind: 'button', button: 'skipRecommended' }
        }

        if (event.x >= layout.primaryButtonStartColumn && event.x <= layout.primaryButtonEndColumn) {
            return { kind: 'button', button: layout.primaryButton }
        }
    }

    return null
}

export function calculateSettingsEditLayout ({
    height,
    optionCount,
    text = getTuiText(),
    terminalColumns,
}: {
    height: number
    optionCount: number
    text?: TuiText
    terminalColumns: number
}): SettingsEditLayout {
    const rootContentWidth = getRootContentWidth(terminalColumns)
    const availableWidth = Math.max(1, rootContentWidth - SETTINGS_DIALOG_SIDE_MARGIN_COLUMNS * 2)
    const dialogWidth = availableWidth < SETTINGS_EDIT_DIALOG_MIN_WIDTH
        ? availableWidth
        : Math.min(SETTINGS_EDIT_DIALOG_MAX_WIDTH, availableWidth)
    const optionRows = getSettingsEditOptionRows(optionCount)
    const inputRows = getSettingsEditInputRows(optionCount)
    const outerRows = SETTINGS_DIALOG_BORDER_ROWS
        + SETTINGS_DIALOG_VERTICAL_PADDING_ROWS
        + SETTINGS_EDIT_DIALOG_HEADER_ROWS
        + SETTINGS_EDIT_DIALOG_INPUT_MARGIN_TOP_ROWS
        + inputRows
        + SETTINGS_EDIT_DIALOG_BUTTON_MARGIN_TOP_ROWS
        + SETTINGS_EDIT_DIALOG_BUTTON_ROWS
    const startColumn = ROOT_PADDING_COLUMNS + 1 + getCenteredOffset(rootContentWidth, dialogWidth)
    const startRow = HEADER_ROWS + 1 + getCenteredOffset(height, outerRows)
    const buttonStartColumn = startColumn + 2
    const [confirmButton, cancelButton] = getButtonRowBounds(buttonStartColumn, getDialogContentWidth(dialogWidth), [
        text.taskDialog.ok,
        text.taskDialog.cancel,
    ])
    const optionStartRow = startRow
        + SETTINGS_DIALOG_BORDER_ROWS / 2
        + SETTINGS_DIALOG_VERTICAL_PADDING_ROWS / 2
        + SETTINGS_EDIT_DIALOG_HEADER_ROWS
        + SETTINGS_EDIT_DIALOG_INPUT_MARGIN_TOP_ROWS
    const buttonRow = optionStartRow + inputRows + SETTINGS_EDIT_DIALOG_BUTTON_MARGIN_TOP_ROWS

    return {
        startColumn,
        endColumn: startColumn + dialogWidth - 1,
        startRow,
        endRow: startRow + outerRows - 1,
        optionStartRow,
        optionEndRow: optionStartRow + optionRows - 1,
        buttonRow,
        confirmButtonStartColumn: confirmButton.startColumn,
        confirmButtonEndColumn: confirmButton.endColumn,
        cancelButtonStartColumn: cancelButton.startColumn,
        cancelButtonEndColumn: cancelButton.endColumn,
    }
}

export function getSettingsEditMouseTarget (
    event: Pick<MouseInputEvent, 'x'|'y'>,
    layout: SettingsEditLayout,
): SettingsEditMouseTarget|null {
    if (event.x < layout.startColumn || event.x > layout.endColumn || event.y < layout.startRow || event.y > layout.endRow) {
        return null
    }

    if (event.y >= layout.optionStartRow && event.y <= layout.optionEndRow) {
        return { kind: 'option', index: event.y - layout.optionStartRow }
    }

    if (event.y === layout.buttonRow) {
        if (event.x >= layout.confirmButtonStartColumn && event.x <= layout.confirmButtonEndColumn) {
            return { kind: 'confirm' }
        }

        if (event.x >= layout.cancelButtonStartColumn && event.x <= layout.cancelButtonEndColumn) {
            return { kind: 'cancel' }
        }
    }

    return null
}

function getSettingsButtonIndex (button: SettingsButton): number {
    return button === 'save' ? 0 : 1
}

function getSettingsButtonByIndex (index: number): SettingsButton {
    return index === 0 ? 'save' : 'back'
}

function clampIndex (index: number, count: number): number {
    const safeCount = Math.max(1, count)

    return Math.min(Math.max(0, index), safeCount - 1)
}

function getSetupWizardPrimaryButton (stepIndex: number): SetupWizardPrimaryButton {
    if (SETUP_WIZARD_STEPS[stepIndex]?.id === 'recommended') {
        return 'applyRecommended'
    }

    return stepIndex >= SETUP_WIZARD_STEPS.length - 1 ? 'finish' : 'next'
}

function getSetupWizardPrimaryButtonLabel (button: SetupWizardPrimaryButton, text: TuiText): string {
    if (button === 'finish') {
        return text.setupWizard.finish
    }

    if (button === 'applyRecommended') {
        return text.setupWizard.applyRecommended
    }

    return text.setupWizard.next
}

function layoutPreviousButtonEnabled (stepIndex: number): boolean {
    return stepIndex > 0
}

function getFocusedSetupWizardButton (focus: SetupWizardButtonFocus, stepIndex: number): SetupWizardButton {
    if (focus === 'previous') {
        return 'previous'
    }

    if (focus === 'retest') {
        return 'retest'
    }

    if (focus === 'secondary') {
        return 'skipRecommended'
    }

    return getSetupWizardPrimaryButton(stepIndex)
}

function getSetupWizardButtonFocusOrder (stepIndex: number): SetupWizardButtonFocus[] {
    const order: SetupWizardButtonFocus[] = layoutPreviousButtonEnabled(stepIndex)
        ? ['previous']
        : []

    if (SETUP_WIZARD_STEPS[stepIndex]?.id === 'apiTest') {
        order.push('retest')
    }

    if (SETUP_WIZARD_STEPS[stepIndex]?.id === 'recommended') {
        order.push('secondary')
    }

    order.push('primary')

    return order
}

function getNextSetupWizardButtonFocus (
    currentFocus: SetupWizardButtonFocus,
    stepIndex: number,
    delta: number,
): SetupWizardButtonFocus {
    const order = getSetupWizardButtonFocusOrder(stepIndex)
    const currentIndex = Math.max(0, order.indexOf(currentFocus))

    return order[(currentIndex + order.length + delta) % order.length] ?? 'primary'
}

export function shouldActivateSetupWizardMouseEvent (event: Pick<MouseInputEvent, 'kind'|'button'>): boolean {
    return event.kind === 'press' && event.button === 'left'
}

function getSetupWizardStaticContentRows (
    stepId: SetupWizardStepId,
    text: TuiText,
    apiTest: SetupWizardApiTestState,
): number {
    return Math.max(1, getSetupWizardStaticLines(stepId, text, apiTest).length)
}

function getSetupWizardVisibleRows (text: TuiText): number {
    const maxBodyContentRows = Math.max(...SETUP_WIZARD_STEPS.map(step => {
        if (step.fields.length > 0) {
            return step.fields.length
        }

        if (step.id === 'apiTest') {
            return getSetupWizardStaticContentRows(step.id, text, { status: 'succeeded', response: '' })
        }

        return getSetupWizardStaticContentRows(step.id, text, { status: 'idle' })
    }))

    return Math.max(SETUP_WIZARD_STEPS.length, SETUP_WIZARD_STEP_HEADER_ROWS + maxBodyContentRows)
}

function getSetupWizardStaticLines (
    stepId: SetupWizardStepId,
    text: TuiText,
    apiTest: SetupWizardApiTestState,
): string[] {
    if (stepId === 'welcome') {
        return text.setupWizard.welcomeLines
    }

    if (stepId === 'recommended') {
        return text.setupWizard.recommendedLines
    }

    if (stepId === 'apiTest' && apiTest.status === 'running') {
        return [
            text.setupWizard.apiTestRunning,
            text.setupWizard.apiTestContinue,
        ]
    }

    if (stepId === 'apiTest' && apiTest.status === 'succeeded') {
        return [
            text.setupWizard.apiTestSucceeded,
            text.setupWizard.apiTestResponse(apiTest.response || text.message.emptyResponse),
        ]
    }

    if (stepId === 'apiTest' && apiTest.status === 'failed') {
        return [
            text.setupWizard.apiTestFailed,
            text.setupWizard.apiTestError(apiTest.error),
        ]
    }

    if (stepId === 'apiTest') {
        return [
            text.setupWizard.apiTestIdle,
            text.setupWizard.apiTestContinue,
        ]
    }

    return []
}

function getSetupWizardStaticLineColor (
    stepId: SetupWizardStepId,
    apiTest: SetupWizardApiTestState,
): string|undefined {
    if (stepId !== 'apiTest') {
        return undefined
    }

    if (apiTest.status === 'succeeded') {
        return 'green'
    }

    if (apiTest.status === 'running') {
        return 'yellow'
    }

    if (apiTest.status === 'failed') {
        return 'red'
    }

    return undefined
}

function getSetupWizardFooterColor (saveMessage: string|null, text: TuiText): string|undefined {
    if (!saveMessage) {
        return undefined
    }

    if (saveMessage === text.setupWizard.saved || saveMessage === text.setupWizard.apiTestSucceeded) {
        return 'green'
    }

    if (saveMessage === text.setupWizard.apiTestRunning) {
        return 'yellow'
    }

    return 'red'
}

function createTaskLogFilesUpdate (root: string, logger: RunLogger): { run_log: string|null, debug_log: string|null } {
    return {
        run_log: logger.filePath ? path.relative(root, logger.filePath) : null,
        debug_log: logger.debugFilePath ? path.relative(root, logger.debugFilePath) : null,
    }
}

export function shouldInterruptPendingGlossarySelection (selection: PendingGlossarySelection|null): selection is PendingGlossarySelection {
    return selection !== null && selection.task.status !== 'completed'
}

function renderButtonLabel (label: string, selected: boolean): React.ReactElement {
    return (
        <Text color={selected ? 'black' : 'cyan'} backgroundColor={selected ? 'cyan' : undefined} bold={selected}>
            {`[ ${label} ]`}
        </Text>
    )
}

function getTaskStatusColor (task: ProjectTask): string {
    return task.status === 'completed'
        ? 'green'
        : task.status === 'failed'
            ? 'red'
            : task.status === 'interrupted'
                ? 'yellow'
                : 'cyan'
}

function formatTaskCardLabel (task: ProjectTask, contentWidth: number, text: TuiText = getTuiText()): string {
    const updatedAt = task.updated_at ? task.updated_at.replace('T', ' ').slice(0, 16) : '-'
    const label = `${task.task_code} | ${formatTaskStatus(task.status, text)}/${formatTaskStage(task.stage, text)} | ${updatedAt}`

    return truncateTerminalText(label, contentWidth)
}

function formatTaskStatus (status: TaskStatus, text: TuiText): string {
    return text.states.taskStatus[status]
}

function formatTaskStage (stage: TaskStage, text: TuiText): string {
    return text.states.taskStage[stage]
}

function getTaskSelectionDialogOuterRows (visibleTaskCount: number): number {
    const choiceCount = Math.min(Math.max(0, visibleTaskCount), MAX_VISIBLE_TASK_CHOICES) + 1

    return TASK_DIALOG_BORDER_ROWS
        + TASK_DIALOG_VERTICAL_PADDING_ROWS
        + TASK_DIALOG_HEADER_ROWS
        + TASK_DIALOG_OPTION_MARGIN_TOP_ROWS
        + choiceCount
        + TASK_DIALOG_FOOTER_ROWS
}

function getStartMenuDialogOuterRows (): number {
    const optionRows = START_MENU_CHOICES.length
    const optionGapRows = Math.max(0, optionRows - 1) * START_MENU_OPTION_GAP_ROWS

    return START_MENU_BORDER_ROWS
        + START_MENU_VERTICAL_PADDING_ROWS
        + START_MENU_HEADER_ROWS
        + START_MENU_OPTION_MARGIN_TOP_ROWS
        + optionRows
        + optionGapRows
}

function getStartMenuChoiceStartRow (startRow: number): number {
    return startRow
        + START_MENU_BORDER_ROWS / 2
        + START_MENU_VERTICAL_PADDING_ROWS / 2
        + START_MENU_HEADER_ROWS
        + START_MENU_OPTION_MARGIN_TOP_ROWS
}

function getStartMenuChoiceRowSpan (): number {
    return 1 + START_MENU_OPTION_GAP_ROWS
}

function getConfigMismatchDialogOuterRows (visibleDifferenceCount: number): number {
    return CONFIG_MISMATCH_BORDER_ROWS
        + CONFIG_MISMATCH_VERTICAL_PADDING_ROWS
        + CONFIG_MISMATCH_HEADER_ROWS
        + CONFIG_MISMATCH_DIFFERENCE_MARGIN_TOP_ROWS
        + Math.max(CONFIG_MISMATCH_MIN_DIFFERENCE_ROWS, visibleDifferenceCount)
        + CONFIG_MISMATCH_BUTTON_MARGIN_TOP_ROWS
        + CONFIG_MISMATCH_BUTTON_ROWS
        + CONFIG_MISMATCH_FOOTER_ROWS
}

function getTaskCompleteDialogOuterRows (): number {
    return TASK_COMPLETE_BORDER_ROWS
        + TASK_COMPLETE_VERTICAL_PADDING_ROWS
        + TASK_COMPLETE_HEADER_ROWS
        + TASK_COMPLETE_BUTTON_MARGIN_TOP_ROWS
        + TASK_COMPLETE_BUTTON_ROWS
        + TASK_COMPLETE_FOOTER_ROWS
}

function getConfigMismatchDifferenceStartRow (startRow: number): number {
    return startRow
        + CONFIG_MISMATCH_BORDER_ROWS / 2
        + CONFIG_MISMATCH_VERTICAL_PADDING_ROWS / 2
        + CONFIG_MISMATCH_HEADER_ROWS
        + CONFIG_MISMATCH_DIFFERENCE_MARGIN_TOP_ROWS
}

function getVisibleConfigMismatchDifferenceCount (differenceCount: number, height: number): number {
    return Math.max(
        CONFIG_MISMATCH_MIN_DIFFERENCE_ROWS,
        Math.min(
            CONFIG_MISMATCH_MAX_VISIBLE_DIFFERENCES,
            differenceCount,
            height - CONFIG_MISMATCH_VISIBLE_DIFFERENCE_RESERVED_ROWS,
        ),
    )
}

function getConfigMismatchDifferenceRows (differenceCount: number, height: number): number {
    const visibleDifferenceCount = getVisibleConfigMismatchDifferenceCount(differenceCount, height)

    return visibleDifferenceCount + (differenceCount > visibleDifferenceCount ? 1 : 0)
}

function getVisibleConfigMismatchDifferences (differences: TaskConfigDifference[], height: number): TaskConfigDifference[] {
    return differences.slice(0, getVisibleConfigMismatchDifferenceCount(differences.length, height))
}

function getRootContentWidth (terminalColumns: number): number {
    return Math.max(1, terminalColumns - ROOT_PADDING_COLUMNS * 2)
}

function getStartMenuDialogWidth (terminalColumns: number): number {
    return Math.min(Math.max(1, getRootContentWidth(terminalColumns) - 2), START_MENU_DIALOG_WIDTH)
}

function getDialogContentWidth (dialogWidth: number): number {
    return Math.max(1, dialogWidth - 4)
}

function getTaskSelectionDialogWidth (terminalColumns: number): number {
    const availableWidth = Math.max(1, getRootContentWidth(terminalColumns) - TASK_DIALOG_SIDE_MARGIN_COLUMNS * 2)

    return availableWidth < TASK_DIALOG_MIN_WIDE_WIDTH
        ? availableWidth
        : Math.min(TASK_DIALOG_MAX_WIDTH, availableWidth)
}

function getCenteredOffset (availableSize: number, childSize: number): number {
    return Math.ceil(Math.max(0, availableSize - childSize) / 2)
}

function getSettingsEditOptionRows (optionCount: number): number {
    return Math.min(optionCount, SETTINGS_EDIT_MAX_VISIBLE_OPTIONS)
}

function getSettingsEditInputRows (optionCount: number): number {
    return Math.max(1, getSettingsEditOptionRows(optionCount))
}

type PaneScrollTargetOptions = {
    panes?: Array<Pick<GlossaryWorkerPaneUpdate, 'slotIndex'>>
    scrollOffsetFromTop?: number
    scrollViewportRows?: number
    showReviewPane?: boolean
    showWorkerPanes?: boolean
}

export function getPaneScrollTarget (
    event: MouseWheelEvent,
    layout: DualPaneLayout,
    options: PaneScrollTargetOptions = {},
): PaneScrollTarget {
    if (event.x === null || event.y === null) {
        return { kind: 'transcript' }
    }

    if (event.y < layout.contentStartRow || event.y > layout.contentEndRow) {
        return { kind: 'transcript' }
    }

    if (event.x < layout.leftStartColumn || event.x > layout.leftEndColumn) {
        return { kind: 'transcript' }
    }

    if (options.showReviewPane) {
        return { kind: 'review-pane' }
    }

    if (options.showWorkerPanes) {
        const slotIndex = getWorkerPaneSlotAtPosition({
            layout,
            panes: options.panes ?? [],
            scrollOffsetFromTop: options.scrollOffsetFromTop ?? 0,
            scrollViewportRows: options.scrollViewportRows,
            x: event.x,
            y: event.y,
        })

        if (slotIndex !== null) {
            return {
                kind: 'worker-pane',
                slotIndex,
            }
        }
    }

    return { kind: 'left-pane' }
}

export function getWorkerPanesPerRow (width: number): number {
    const occupiedWidth = WORKER_PANE_MIN_WIDTH + WORKER_PANE_MARGIN_RIGHT

    return Math.max(1, Math.floor((Math.max(1, width) + WORKER_PANE_MARGIN_RIGHT) / occupiedWidth))
}

export function calculateWorkerPaneGridLayout (panes: Array<Pick<GlossaryWorkerPaneUpdate, 'slotIndex'>>, width: number): WorkerPaneGridLayout {
    const contentWidth = Math.max(1, width)
    const paneCount = Math.max(1, panes.length)
    const maxDenseColumns = Math.max(1, Math.ceil(Math.sqrt(paneCount)))
    const panesPerRow = Math.min(paneCount, maxDenseColumns, getWorkerPanesPerRow(contentWidth))
    const totalMarginWidth = panesPerRow * WORKER_PANE_MARGIN_RIGHT
    const paneWidth = Math.max(1, Math.floor((contentWidth - totalMarginWidth) / panesPerRow))
    const rowCount = panes.length === 0 ? 0 : Math.ceil(panes.length / panesPerRow)

    return {
        contentRows: panes.length === 0
            ? WORKER_GRID_TITLE_ROWS + 1
            : WORKER_GRID_TITLE_ROWS + rowCount * (WORKER_PANE_HEIGHT + WORKER_PANE_MARGIN_BOTTOM),
        contentWidth,
        hasScrollBar: false,
        paneWidth,
        panesPerRow,
    }
}

export function calculateScrollableWorkerPaneGridLayout (panes: Array<Pick<GlossaryWorkerPaneUpdate, 'slotIndex'>>, width: number, viewportRows: number): WorkerPaneGridLayout {
    const fullWidthLayout = calculateWorkerPaneGridLayout(panes, width)

    if (fullWidthLayout.contentRows <= Math.max(1, viewportRows)) {
        return fullWidthLayout
    }

    return {
        ...calculateWorkerPaneGridLayout(panes, Math.max(1, width - 1)),
        hasScrollBar: true,
    }
}

export function getWorkerPaneSlotAtPosition ({
    layout,
    panes,
    scrollOffsetFromTop,
    scrollViewportRows,
    x,
    y,
}: {
    layout: DualPaneLayout
    panes: Array<Pick<GlossaryWorkerPaneUpdate, 'slotIndex'>>
    scrollOffsetFromTop: number
    scrollViewportRows?: number
    x: number
    y: number
}): number|null {
    const localX = x - layout.leftStartColumn
    const localY = y - layout.contentStartRow + scrollOffsetFromTop

    if (localX < 0 || localY < WORKER_GRID_TITLE_ROWS) {
        return null
    }

    const gridLayout = scrollViewportRows === undefined
        ? calculateWorkerPaneGridLayout(panes, layout.paneWidth)
        : calculateScrollableWorkerPaneGridLayout(panes, layout.paneWidth, scrollViewportRows)
    const rowSpan = WORKER_PANE_HEIGHT + WORKER_PANE_MARGIN_BOTTOM
    const paneRow = Math.floor((localY - WORKER_GRID_TITLE_ROWS) / rowSpan)
    const rowOffset = (localY - WORKER_GRID_TITLE_ROWS) % rowSpan

    if (rowOffset >= WORKER_PANE_HEIGHT) {
        return null
    }

    const columnSpan = gridLayout.paneWidth + WORKER_PANE_MARGIN_RIGHT
    const paneColumn = Math.floor(localX / columnSpan)
    const columnOffset = localX % columnSpan

    if (columnOffset >= gridLayout.paneWidth || paneColumn >= gridLayout.panesPerRow) {
        return null
    }

    const paneIndex = paneRow * gridLayout.panesPerRow + paneColumn

    return panes[paneIndex]?.slotIndex ?? null
}

export function parseMouseWheelDirections (value: string): Array<'up'|'down'> {
    return parseMouseWheelEvents(value).map(event => event.direction)
}

export function parseMouseWheelEvents (value: string): MouseWheelEvent[] {
    const events: MouseWheelEvent[] = []
    const sgrMousePattern = /(?:\u001B)?\[<(\d+);(\d+);(\d+)[mM]/gu
    let match: RegExpExecArray|null

    while ((match = sgrMousePattern.exec(value)) !== null) {
        const direction = getWheelDirection(Number(match[1]))

        if (direction) {
            events.push({
                direction,
                x: Number(match[2]),
                y: Number(match[3]),
            })
        }
    }

    const x10MousePattern = /(?:\u001B)?\[M([\s\S])([\s\S])([\s\S])/gu

    while ((match = x10MousePattern.exec(value)) !== null) {
        const direction = getWheelDirection(match[1].charCodeAt(0) - 32)

        if (direction) {
            events.push({
                direction,
                x: match[2].charCodeAt(0) - 32,
                y: match[3].charCodeAt(0) - 32,
            })
        }
    }

    const urxvtMousePattern = /(?:\u001B)?\[(\d+);(\d+);(\d+)M/gu

    while ((match = urxvtMousePattern.exec(value)) !== null) {
        const direction = getWheelDirection(Number(match[1]))

        if (direction) {
            events.push({
                direction,
                x: Number(match[2]),
                y: Number(match[3]),
            })
        }
    }

    return events
}

export function parseMouseEvents (value: string): MouseInputEvent[] {
    const events: MouseInputEvent[] = []
    const sgrMousePattern = /(?:\u001B)?\[<(\d+);(\d+);(\d+)([mM])/gu
    let match: RegExpExecArray|null

    while ((match = sgrMousePattern.exec(value)) !== null) {
        const buttonCode = Number(match[1])
        const direction = getWheelDirection(buttonCode)
        const x = Number(match[2])
        const y = Number(match[3])
        const finalByte = match[4]

        if (direction) {
            events.push({
                kind: 'wheel',
                button: null,
                direction,
                x,
                y,
            })
            continue
        }

        const kind = finalByte === 'm'
            ? 'release'
            : buttonCode & 32
                ? 'move'
                : 'press'

        events.push({
            kind,
            button: (buttonCode & 3) === 0 ? 'left' : 'other',
            x,
            y,
        })
    }

    const x10MousePattern = /(?:\u001B)?\[M([\s\S])([\s\S])([\s\S])/gu

    while ((match = x10MousePattern.exec(value)) !== null) {
        const buttonCode = match[1].charCodeAt(0) - 32
        const direction = getWheelDirection(buttonCode)
        const x = match[2].charCodeAt(0) - 32
        const y = match[3].charCodeAt(0) - 32

        if (direction) {
            events.push({
                kind: 'wheel',
                button: null,
                direction,
                x,
                y,
            })
            continue
        }

        events.push({
            kind: getMouseButtonKind(buttonCode),
            button: getMouseButton(buttonCode),
            x,
            y,
        })
    }

    const urxvtMousePattern = /(?:\u001B)?\[(\d+);(\d+);(\d+)M/gu

    while ((match = urxvtMousePattern.exec(value)) !== null) {
        const buttonCode = Number(match[1])
        const direction = getWheelDirection(buttonCode)
        const x = Number(match[2])
        const y = Number(match[3])

        if (direction) {
            events.push({
                kind: 'wheel',
                button: null,
                direction,
                x,
                y,
            })
            continue
        }

        events.push({
            kind: getMouseButtonKind(buttonCode),
            button: getMouseButton(buttonCode),
            x,
            y,
        })
    }

    return events
}

function getMouseButtonKind (buttonCode: number): MouseInputEvent['kind'] {
    if ((buttonCode & 3) === 3) {
        return 'release'
    }

    return buttonCode & 32 ? 'move' : 'press'
}

function getMouseButton (buttonCode: number): MouseInputEvent['button'] {
    return (buttonCode & 3) === 0 ? 'left' : 'other'
}

export function parseMousePositionEvents (value: string): MousePositionEvent[] {
    const events: MousePositionEvent[] = []
    const sgrMousePattern = /(?:\u001B)?\[<\d+;(\d+);(\d+)[mM]/gu
    let match: RegExpExecArray|null

    while ((match = sgrMousePattern.exec(value)) !== null) {
        events.push({
            x: Number(match[1]),
            y: Number(match[2]),
        })
    }

    const x10MousePattern = /(?:\u001B)?\[M[\s\S]([\s\S])([\s\S])/gu

    while ((match = x10MousePattern.exec(value)) !== null) {
        events.push({
            x: match[1].charCodeAt(0) - 32,
            y: match[2].charCodeAt(0) - 32,
        })
    }

    const urxvtMousePattern = /(?:\u001B)?\[\d+;(\d+);(\d+)M/gu

    while ((match = urxvtMousePattern.exec(value)) !== null) {
        events.push({
            x: Number(match[1]),
            y: Number(match[2]),
        })
    }

    return events
}

function getWheelDirection (buttonCode: number): 'up'|'down'|null {
    const wheelCode = buttonCode & 67

    if (wheelCode === 64) {
        return 'up'
    }

    if (wheelCode === 65) {
        return 'down'
    }

    return null
}

export function getKeyboardScrollDelta (
    inputValue: string,
    key: { pageUp?: boolean, pageDown?: boolean, upArrow?: boolean, downArrow?: boolean },
    pageSize: number,
): number {
    if (key.pageUp || isPageUpInput(inputValue)) {
        return Math.max(1, pageSize)
    }

    if (key.pageDown || isPageDownInput(inputValue)) {
        return -Math.max(1, pageSize)
    }

    if (key.upArrow || isUpArrowInput(inputValue)) {
        return 1
    }

    if (key.downArrow || isDownArrowInput(inputValue)) {
        return -1
    }

    return 0
}

export function getTaskSelectionDelta (
    inputValue: string,
    key: { upArrow?: boolean, downArrow?: boolean },
): number {
    if (key.upArrow || isUpArrowInput(inputValue) || inputValue === 'k') {
        return -1
    }

    if (key.downArrow || isDownArrowInput(inputValue) || inputValue === 'j') {
        return 1
    }

    return 0
}

export function getNextTaskSelectionKeyboardTarget (
    current: TaskSelectionMouseTarget|null,
    selectedIndex: number,
): TaskSelectionMouseTarget|null {
    void selectedIndex
    const targets: Array<TaskSelectionMouseTarget|null> = [
        null,
        { kind: 'confirm' },
        { kind: 'cancel' },
        { kind: 'change-directory' },
    ]
    const normalizedCurrent = current?.kind === 'choice' ? null : current
    const currentIndex = targets.findIndex(target => isSameNullableTaskSelectionTarget(target, normalizedCurrent))

    return targets[(currentIndex + 1) % targets.length] ?? null
}

export function getNextTaskSelectionButtonTarget (
    current: TaskSelectionButtonTarget|null,
    step: number,
): TaskSelectionButtonTarget {
    const targets: TaskSelectionButtonTarget[] = [
        { kind: 'confirm' },
        { kind: 'cancel' },
        { kind: 'change-directory' },
    ]
    const currentIndex = current ? targets.findIndex(target => target.kind === current.kind) : -1

    return targets[(currentIndex + targets.length + step) % targets.length] ?? targets[0]!
}

export function getNextSettingsEditKeyboardTarget (
    current: SettingsEditMouseTarget|null,
    selectedOptionIndex: number,
    optionCount: number,
): SettingsEditMouseTarget|null {
    void selectedOptionIndex
    void optionCount
    const targets: Array<SettingsEditMouseTarget|null> = [
        null,
        { kind: 'confirm' },
        { kind: 'cancel' },
    ]
    const normalizedCurrent = current?.kind === 'option' ? null : current
    const currentIndex = targets.findIndex(target => isSameSettingsEditTarget(target, normalizedCurrent))

    return targets[(currentIndex + 1) % targets.length] ?? null
}

export function getNextSettingsEditButtonTarget (
    current: Extract<SettingsEditMouseTarget, { kind: 'confirm'|'cancel' }>|null,
    step: number,
): Extract<SettingsEditMouseTarget, { kind: 'confirm'|'cancel' }> {
    const targets: Array<Extract<SettingsEditMouseTarget, { kind: 'confirm'|'cancel' }>> = [
        { kind: 'confirm' },
        { kind: 'cancel' },
    ]
    const currentIndex = current ? targets.findIndex(target => target.kind === current.kind) : -1

    return targets[(currentIndex + targets.length + step) % targets.length] ?? targets[0]!
}

function isTaskSelectionButtonTarget (target: TaskSelectionMouseTarget|null): target is TaskSelectionButtonTarget {
    return target !== null && target.kind !== 'choice'
}

function isSameTaskSelectionTarget (left: TaskSelectionMouseTarget, right: TaskSelectionMouseTarget): boolean {
    if (left.kind !== right.kind) {
        return false
    }

    return left.kind !== 'choice' || right.kind !== 'choice' || left.index === right.index
}

function isSameNullableTaskSelectionTarget (left: TaskSelectionMouseTarget|null, right: TaskSelectionMouseTarget|null): boolean {
    if (left === null || right === null) {
        return left === right
    }

    return isSameTaskSelectionTarget(left, right)
}

function isSameSettingsEditTarget (left: SettingsEditMouseTarget|null, right: SettingsEditMouseTarget|null): boolean {
    if (left === null || right === null) {
        return left === right
    }

    if (left.kind !== right.kind) {
        return false
    }

    return left.kind !== 'option' || right.kind !== 'option' || left.index === right.index
}

function isMouseInput (value: string): boolean {
    return parseMouseWheelEvents(value).length > 0
        || parseMouseEvents(value).length > 0
        || parseMousePositionEvents(value).length > 0
        || value.startsWith('[<64;')
        || value.startsWith('[<65;')
        || value.startsWith('[64;')
        || value.startsWith('[65;')
        || value.startsWith('[M`')
        || value.startsWith('[Ma')
        || value.startsWith('\u001B[<64;')
        || value.startsWith('\u001B[<65;')
        || value.startsWith('\u001B[64;')
        || value.startsWith('\u001B[65;')
        || value.startsWith('\u001B[M`')
        || value.startsWith('\u001B[Ma')
}

function isPageUpInput (value: string): boolean {
    return value === '[5~'
        || value === '[[5~'
        || value === '[5$'
        || value === '[5^'
        || value.startsWith('[5;')
        || value.startsWith('\u001B[5~')
        || value.startsWith('\u001B[[5~')
        || value.startsWith('\u001B[5$')
        || value.startsWith('\u001B[5^')
        || value.startsWith('\u001B[5;')
}

function isPageDownInput (value: string): boolean {
    return value === '[6~'
        || value === '[[6~'
        || value === '[6$'
        || value === '[6^'
        || value.startsWith('[6;')
        || value.startsWith('\u001B[6~')
        || value.startsWith('\u001B[[6~')
        || value.startsWith('\u001B[6$')
        || value.startsWith('\u001B[6^')
        || value.startsWith('\u001B[6;')
}

function isUpArrowInput (value: string): boolean {
    return value === '[A' || value === 'OA' || value.startsWith('[1;') && value.endsWith('A')
}

function isDownArrowInput (value: string): boolean {
    return value === '[B' || value === 'OB' || value.startsWith('[1;') && value.endsWith('B')
}

function getTranscriptStyle (entry: TranscriptEntry, text: TuiText): { color: string, label: string } {
    const color = entry.kind === 'user'
        ? 'green'
        : entry.kind === 'assistant'
            ? 'white'
            : entry.kind === 'tool'
                ? 'yellow'
                : entry.kind === 'error'
                    ? 'red'
                    : 'gray'

    const label = entry.kind === 'user'
        ? text.transcript.user
        : entry.kind === 'assistant'
            ? text.transcript.assistant
            : entry.kind === 'tool'
                ? text.transcript.tool
                : entry.kind === 'error'
                    ? text.transcript.error
                    : text.transcript.system

    return { color, label }
}

function DisplayLineText ({ line }: { line: DisplayLine }): React.ReactElement {
    return (
        <Text color={line.color} bold={line.bold} dimColor={line.dimColor} wrap="truncate-end">
            {line.text}
        </Text>
    )
}

function PaneLineText ({ line }: { line: PaneLine }): React.ReactElement {
    return (
        <Text color={line.color} bold={line.bold} dimColor={line.dimColor} wrap="truncate">
            {line.text}
        </Text>
    )
}

function OptionalScrollBar ({
    color,
    contentHeight,
    scrollOffset,
    viewportHeight,
}: {
    color?: string
    contentHeight: number
    scrollOffset: number
    viewportHeight: number
}): React.ReactElement|null {
    if (contentHeight <= viewportHeight) {
        return null
    }

    return (
        <ScrollBar
            autoHide
            color={color}
            contentHeight={contentHeight}
            placement="inset"
            scrollOffset={scrollOffset}
            style="line"
            viewportHeight={viewportHeight}
        />
    )
}

export function getTranscriptTopScrollOffset (lineCount: number, viewportHeight: number, scrollOffsetFromBottom: number): number {
    return getMaxScrollOffset(lineCount, viewportHeight) - clampScrollOffset(scrollOffsetFromBottom, lineCount, viewportHeight)
}

function TranscriptPane ({
    height,
    lines,
    marginLeft = 0,
    maxScrollOffset,
    scrollOffsetFromBottom,
    width,
}: {
    height: number
    lines: DisplayLine[]
    marginLeft?: number
    maxScrollOffset: number
    scrollOffsetFromBottom: number
    width: number
}): React.ReactElement {
    const [measuredContentHeight, setMeasuredContentHeight] = useState(0)
    const viewportHeight = Math.max(1, height)
    const contentHeight = measuredContentHeight > 0 ? measuredContentHeight : lines.length
    const scrollOffset = measuredContentHeight > 0
        ? Math.max(0, contentHeight - viewportHeight - clampScrollOffset(scrollOffsetFromBottom, contentHeight, viewportHeight))
        : 0
    const showScrollBar = contentHeight > viewportHeight
    const contentWidth = showScrollBar ? Math.max(1, width - 1) : width

    return (
        <Box flexDirection="row" height={viewportHeight} marginLeft={marginLeft} overflowY="hidden" width={width}>
            <ControlledScrollView
                height={viewportHeight}
                onContentHeightChange={height => {
                    setMeasuredContentHeight(height)
                }}
                scrollOffset={scrollOffset}
                width={contentWidth}
            >
                {lines.map(line => (
                    <DisplayLineText key={line.id} line={line} />
                ))}
            </ControlledScrollView>
            <OptionalScrollBar
                contentHeight={contentHeight}
                scrollOffset={(measuredContentHeight > 0 ? Math.max(0, contentHeight - viewportHeight) : maxScrollOffset) - clampScrollOffset(scrollOffsetFromBottom, contentHeight, viewportHeight)}
                viewportHeight={viewportHeight}
            />
        </Box>
    )
}

function ScrollableReviewPane ({
    height,
    pane,
    paneScrollOffset,
    text,
    width,
}: {
    height: number
    pane: ReviewPaneState
    paneScrollOffset: number
    text: TuiText
    width: number
}): React.ReactElement {
    return (
        <Box flexDirection="column" height={height} overflowY="hidden" width={width}>
            <ReviewPane height={height} pane={pane} scrollOffset={paneScrollOffset} text={text} width={width} />
        </Box>
    )
}

function ScrollableWorkerPaneList ({
    height,
    panes,
    scrollOffset,
    text,
    title,
    width,
    workerPaneScrollOffsets,
}: {
    height: number
    panes: GlossaryWorkerPaneUpdate[]
    scrollOffset: number
    text: TuiText
    title: string
    width: number
    workerPaneScrollOffsets: Record<number, number>
}): React.ReactElement {
    const viewportHeight = Math.max(1, height)
    const scrollLayout = calculateScrollableWorkerPaneGridLayout(panes, width, viewportHeight)

    return (
        <Box flexDirection="row" height={viewportHeight} overflowY="hidden" width={width}>
            <ControlledScrollView height={viewportHeight} overflowX="visible" scrollOffset={scrollOffset} width={scrollLayout.contentWidth}>
                <WorkerPaneGrid layout={scrollLayout} panes={panes} text={text} title={title} workerPaneScrollOffsets={workerPaneScrollOffsets} />
            </ControlledScrollView>
            <OptionalScrollBar
                color="cyan"
                contentHeight={scrollLayout.contentRows}
                scrollOffset={scrollOffset}
                viewportHeight={viewportHeight}
            />
        </Box>
    )
}

function formatReviewPaneLines (pane: ReviewPaneState, width = 120, text: TuiText = getTuiText()): PaneLine[] {
    const contentWidth = getPaneTextWidth(width)
    const batchLabel = pane.batchNumbers.length > 0 ? pane.batchNumbers.join(', ') : '-'
    const lines: PaneLine[] = [
        {
            id: 'review-window',
            text: `${text.pane.window}: ${pane.reviewWindowId ?? '-'}`,
        },
        {
            id: 'review-batches',
            text: `${text.pane.batches}: ${batchLabel} - ${text.pane.pending}: ${pane.pendingBatchCount}`,
        },
        {
            id: 'review-tool',
            text: `${text.pane.tool}: ${pane.lastTool ?? '-'}`,
        },
    ]

    if (pane.error) {
        lines.push(...wrapPaneLine({
            color: 'red',
            id: 'review-error',
            text: `${text.pane.error}: ${normalizePaneText(pane.error)}`,
            width: contentWidth,
        }))
    } else if (pane.summary) {
        lines.push(...wrapPaneLine({
            dimColor: true,
            id: 'review-summary',
            text: `${text.pane.summary}: ${normalizePaneText(pane.summary)}`,
            width: contentWidth,
        }))
    }

    if (pane.events.length > 0) {
        lines.push({
            id: 'review-activity-title',
            text: text.pane.recentReviewActivity,
            color: 'yellow',
            bold: true,
        })

        for (const [eventIndex, event] of pane.events.entries()) {
            for (const [lineIndex, line] of event.split('\n').entries()) {
                lines.push(...wrapPaneLine({
                    id: `review-event-${eventIndex}-${lineIndex}`,
                    text: normalizePaneText(line),
                    dimColor: lineIndex > 0,
                    width: contentWidth,
                }))
            }

            lines.push({
                id: `review-event-${eventIndex}-spacer`,
                text: '',
            })
        }
    }

    return lines
}

function getReviewPaneColor (pane: Pick<ReviewPaneState, 'status'>): string {
    return pane.status === 'failed'
        ? 'red'
        : pane.status === 'completed'
            ? 'green'
            : pane.status === 'idle'
                ? 'gray'
                : 'cyan'
}

function getWorkerPaneColor (pane: Pick<GlossaryWorkerPaneUpdate, 'status'>): string {
    return pane.status === 'failed'
        ? 'red'
        : pane.status === 'completed'
            ? 'green'
            : pane.status === 'running' || pane.status === 'starting'
                ? 'yellow'
                : 'gray'
}

function formatPaneStatus (status: GlossaryWorkerStatus|GlossaryReviewStatus, text: TuiText): string {
    return text.states.paneStatus[status] ?? status
}

function getWorkerPaneViewportRows (): number {
    return Math.max(1, WORKER_PANE_HEIGHT - 2)
}

function getReviewPaneContentRows (pane: ReviewPaneState, width = 120, text: TuiText = getTuiText()): number {
    return formatReviewPaneLines(pane, width, text).length + 1
}

function WorkerPaneGrid ({
    layout,
    panes,
    text,
    title,
    workerPaneScrollOffsets,
}: {
    layout: WorkerPaneGridLayout
    panes: GlossaryWorkerPaneUpdate[]
    text: TuiText
    title: string
    workerPaneScrollOffsets: Record<number, number>
}): React.ReactElement {
    return (
        <Box flexDirection="column" marginBottom={1}>
            <Text color="cyan" bold>
                {title}
            </Text>
            <Box flexDirection="row" flexWrap="wrap">
                {panes.map(pane => (
                    <WorkerPane key={pane.slotIndex} pane={pane} scrollOffset={workerPaneScrollOffsets[pane.slotIndex] ?? 0} text={text} width={layout.paneWidth} />
                ))}
            </Box>
        </Box>
    )
}

function ReviewPane ({
    height,
    pane,
    scrollOffset,
    text,
    width = 120,
}: {
    height: number
    pane: ReviewPaneState
    scrollOffset: number
    text: TuiText
    width?: number
}): React.ReactElement {
    const color = getReviewPaneColor(pane)
    const viewportRows = Math.max(1, height - 2)
    const lines: PaneLine[] = [
        {
            id: 'review-title',
            text: `${text.pane.reviewAgent} - ${formatPaneStatus(pane.status, text)}`,
            color,
            bold: true,
        },
        ...formatReviewPaneLines(pane, width, text),
    ]
    const hasScrollBar = lines.length > viewportRows
    const contentWidth = hasScrollBar ? Math.max(1, width - 5) : Math.max(1, width - 4)

    return (
        <Box
            borderStyle="double"
            borderColor={color}
            flexDirection="column"
            paddingX={1}
            marginBottom={1}
            height={height}
            width={width}
        >
            <Box flexDirection="row" height={viewportRows} overflowY="hidden">
                <ControlledScrollView height={viewportRows} scrollOffset={scrollOffset} width={contentWidth}>
                    {lines.map(line => (
                        <PaneLineText key={line.id} line={line} />
                    ))}
                </ControlledScrollView>
                <OptionalScrollBar
                    color={color}
                    contentHeight={lines.length}
                    scrollOffset={scrollOffset}
                    viewportHeight={viewportRows}
                />
            </Box>
        </Box>
    )
}

function WorkerPane ({ pane, scrollOffset, text, width }: { pane: GlossaryWorkerPaneUpdate, scrollOffset: number, text: TuiText, width: number }): React.ReactElement {
    const color = getWorkerPaneColor(pane)
    const lines = formatWorkerPaneLines(pane, width, text)

    return (
        <Box
            borderStyle="round"
            borderColor={color}
            flexDirection="column"
            height={WORKER_PANE_HEIGHT}
            paddingX={1}
            marginRight={WORKER_PANE_MARGIN_RIGHT}
            marginBottom={WORKER_PANE_MARGIN_BOTTOM}
            overflowY="hidden"
            width={width}
        >
            <ControlledScrollView height={getWorkerPaneViewportRows()} scrollOffset={scrollOffset} width={getPaneTextWidth(width)}>
                {lines.map(line => (
                    <PaneLineText key={line.id} line={line} />
                ))}
            </ControlledScrollView>
        </Box>
    )
}

export function formatWorkerPaneLines (pane: GlossaryWorkerPaneUpdate, paneWidth = WORKER_PANE_MIN_WIDTH, text: TuiText = getTuiText()): PaneLine[] {
    const color = getWorkerPaneColor(pane)
    const batchLabel = pane.batchId
        ? `${pane.batchId}${pane.batchNumber ? ` (${pane.batchNumber}/${pane.totalBatches})` : ''}`
        : text.pane.idle
    const rangeLabel = pane.batchStartIndex === null || pane.batchEndIndex === null
        ? '-'
        : `${pane.batchStartIndex}-${pane.batchEndIndex}`
    const lifecycleLabel = pane.status === 'idle'
        ? '-'
        : `${pane.lifecycle}`
    const contentWidth = getPaneTextWidth(paneWidth)
    const lines: PaneLine[] = [
        {
            id: `worker-${pane.slotIndex}-title`,
            text: `${text.pane.worker} ${pane.slotIndex + 1} - ${formatPaneStatus(pane.status, text)}`,
            color,
            bold: true,
        },
        {
            id: `worker-${pane.slotIndex}-lifecycle`,
            text: `${text.pane.lifecycle}: ${lifecycleLabel}`,
        },
        {
            id: `worker-${pane.slotIndex}-batch`,
            text: `${text.pane.batch}: ${batchLabel}`,
        },
        {
            id: `worker-${pane.slotIndex}-filtered`,
            text: `${text.pane.filtered}: ${rangeLabel} - ${text.pane.keys}: ${pane.keyCount}`,
        },
        ...wrapPaneLine({
            id: `worker-${pane.slotIndex}-tool`,
            text: `${text.pane.tool}: ${pane.lastTool ?? '-'}`,
            width: contentWidth,
        }),
    ]

    if (pane.error) {
        lines.push(...wrapPaneLine({
            color: 'red',
            id: `worker-${pane.slotIndex}-error`,
            text: `${text.pane.error}: ${normalizePaneText(pane.error)}`,
            width: contentWidth,
        }))
    } else if (pane.summary) {
        lines.push(...wrapPaneLine({
            dimColor: true,
            id: `worker-${pane.slotIndex}-summary`,
            text: `${text.pane.summary}: ${normalizePaneText(pane.summary)}`,
            width: contentWidth,
        }))
    }

    return lines
}

function formatApiEventForPane (event: AgentApiEvent, text: TuiText): string {
    if (event.status === 'http_response' || event.status === 'http_error') {
        const http = event.http
        const summary = event.status === 'http_response'
            ? text.pane.apiResponse(String(http?.status ?? '-'), http?.contentType ?? '-')
            : text.pane.apiTransportError
        const details = [
            http ? `${http.method} ${http.url}` : '',
            text.pane.stream(http?.requestStream === null ? text.pane.unknown : String(http?.requestStream)),
            http?.requestId ? text.pane.requestId(http.requestId) : '',
            text.pane.duration(http?.durationMs ?? 0),
            event.diagnostic ? text.pane.diagnostic(event.diagnostic) : '',
            event.error?.message ? text.pane.eventError(truncatePaneText(event.error.message)) : '',
        ].filter(Boolean)

        return [summary, ...details].join('\n')
    }

    const attempt = text.pane.attempt(event.attempt, event.maxAttempts)
    const details = [
        event.retryable === undefined ? '' : text.pane.retryable(event.retryable),
        event.delayMs ? text.pane.retryIn(event.delayMs) : '',
        event.diagnostic ? text.pane.diagnostic(event.diagnostic) : '',
        event.error?.message ? text.pane.eventError(truncatePaneText(event.error.message)) : '',
    ].filter(Boolean)

    return [text.pane.apiEvent(event.status, attempt), ...details].join('\n')
}

function StatusLine ({
    apiStatus,
    columns,
    maxScrollOffset,
    scrollOffsetFromBottom,
    stage,
    text,
    tokenPulse,
    tokenUsage,
}: {
    apiStatus: ApiStatusState|null
    columns: number
    maxScrollOffset: number
    scrollOffsetFromBottom: number
    stage: AppStage
    text: TuiText
    tokenPulse: TokenPulseState
    tokenUsage: TokenUsageState
}): React.ReactElement {
    const scrollLabel = maxScrollOffset === 0
        ? text.status.bottom
        : scrollOffsetFromBottom === 0
            ? text.status.bottom
            : `-${scrollOffsetFromBottom}/${maxScrollOffset}`
    const apiLabelWidth = Math.max(12, columns - 76)
    const apiLabel = apiStatus
        ? `${apiStatus.status}${apiStatus.maxAttempts > 0 ? ` ${apiStatus.attempt}/${apiStatus.maxAttempts}` : ''} ${truncateStatusValue(apiStatus.label, apiLabelWidth)}`
        : text.status.idle

    return (
        <Box height={1} overflow="hidden" width="100%" backgroundColor="blue">
            <Text backgroundColor="blue" color="white" wrap="truncate-end">
                {` ${stage} | ${text.status.scroll} ${scrollLabel} | ${text.status.api} ${apiLabel} | `}
                <Text color={tokenPulse.inputDelta > 0 ? 'yellow' : 'white'} bold={tokenPulse.inputDelta > 0}>
                    {`↑ ${formatInteger(tokenUsage.inputTokens)}${formatDelta(tokenPulse.inputDelta)}`}
                </Text>
                {' '}
                <Text color={tokenPulse.outputDelta > 0 ? 'green' : 'white'} bold={tokenPulse.outputDelta > 0}>
                    {`↓ ${formatInteger(tokenUsage.outputTokens)}${formatDelta(tokenPulse.outputDelta)}`}
                </Text>
                {' '}
                <Text color={tokenPulse.totalDelta > 0 ? 'cyan' : 'white'} bold={tokenPulse.totalDelta > 0}>
                    {`${text.status.total} ${formatInteger(tokenUsage.totalTokens)}${formatDelta(tokenPulse.totalDelta)}`}
                </Text>
            </Text>
        </Box>
    )
}

function formatInteger (value: number): string {
    return Math.round(value).toLocaleString('en-US')
}

function formatDelta (value: number): string {
    return value > 0 ? ` +${formatInteger(value)}` : ''
}

function truncateStatusValue (value: string, width: number): string {
    if (getDisplayWidth(value) <= width) {
        return value
    }

    return truncateTerminalText(value, width)
}

function truncatePaneText (value: string): string {
    const normalizedValue = value.replace(/\s+/g, ' ').trim()
    return normalizedValue.length > 120 ? `${normalizedValue.slice(0, 117)}...` : normalizedValue
}

function normalizePaneText (value: string): string {
    return value.replace(/\s+/g, ' ').trim()
}

function getPaneTextWidth (paneWidth: number): number {
    return Math.max(1, paneWidth - 4)
}

function wrapPaneLine ({
    bold,
    color,
    dimColor,
    id,
    text,
    width,
}: {
    bold?: boolean
    color?: string
    dimColor?: boolean
    id: string
    text: string
    width: number
}): PaneLine[] {
    return wrapTextForPane(text, width).map((line, index) => ({
        id: `${id}-${index}`,
        text: line,
        color,
        bold,
        dimColor,
    }))
}

export function wrapTextForPane (value: string, width: number): string[] {
    const safeWidth = Math.max(1, width)
    const normalizedValue = value.length > 0 ? value : ' '
    const lines: string[] = []

    for (const paragraph of normalizedValue.split(/\r\n|\r|\n/g)) {
        lines.push(...wrapAnsi(paragraph, safeWidth, {
            hard: true,
            trim: false,
            wordWrap: false,
        }).split('\n'))
    }

    return lines.length > 0 ? lines : ['']
}

export function truncateForDisplay (value: string, hasRunLog = true, text: TuiText = getTuiText()): string {
    if (value.length <= 600) {
        return value
    }

    return `${value.slice(0, 600)}\n${hasRunLog ? text.message.truncatedWithLog : text.message.truncated}`
}

export function getVisibleTaskChoices (tasks: ProjectTask[], maxCount = MAX_VISIBLE_TASK_CHOICES): ProjectTask[] {
    return tasks.slice(0, Math.max(0, maxCount))
}

export function getTaskSelectionChoiceCount (tasks: ProjectTask[]): number {
    return tasks.length + 1
}

export function getSelectedTaskCode (tasks: ProjectTask[], selectedIndex: number): string|null {
    return selectedIndex >= 0 && selectedIndex < tasks.length
        ? tasks[selectedIndex]?.task_code ?? null
        : null
}

export function getConfigMismatchChoice (inputValue: string, key: { return?: boolean }): ConfigMismatchChoice|null {
    if (key.return || inputValue === '1' || inputValue === 'u') {
        return 'use-task-config'
    }

    return null
}

export function isPasteInput (inputValue: string, key: { ctrl?: boolean }): boolean {
    return key.ctrl === true && inputValue.toLowerCase() === 'v'
}

export function getClipboardPasteContextKey (
    stage: AppStage,
    settingsEdit: Pick<SettingsEditState, 'field'>|null,
): string {
    if (stage === 'settings-edit') {
        return `${stage}:${settingsEdit?.field ?? 'none'}`
    }

    return stage
}

export function isCurrentClipboardPasteContext (current: ClipboardPasteContext, requested: ClipboardPasteContext): boolean {
    return current.key === requested.key && current.version === requested.version
}

export function normalizeClipboardText (value: string): string {
    return value
        .replace(/\u0000/g, '')
        .replace(/\r\n/g, '\n')
        .replace(/\r/g, '\n')
        .replace(/[\u0001-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g, '')
}

type ClipboardReadCommand = {
    command: string
    args: string[]
}

type ClipboardCommandRunner = (command: string, args: string[]) => Promise<string>

export type ReadClipboardTextOptions = {
    platform?: NodeJS.Platform
    runCommand?: ClipboardCommandRunner
}

export async function readClipboardText ({
    platform = process.platform,
    runCommand = runClipboardCommand,
}: ReadClipboardTextOptions = {}): Promise<string|null> {
    for (const { command, args } of getClipboardReadCommands(platform)) {
        try {
            const text = normalizeClipboardText(await runCommand(command, args))

            if (text.length > 0) {
                return text
            }
        } catch {
            // Try the next platform clipboard command.
        }
    }

    return null
}

function getClipboardReadCommands (platform: NodeJS.Platform): ClipboardReadCommand[] {
    if (platform === 'win32') {
        const script = '$text = Get-Clipboard -Raw; if ($null -ne $text) { [Console]::Out.Write($text) }'

        return [
            { command: 'powershell.exe', args: ['-NoProfile', '-NonInteractive', '-Command', script] },
            { command: 'pwsh', args: ['-NoProfile', '-NonInteractive', '-Command', script] },
        ]
    }

    if (platform === 'darwin') {
        return [
            { command: 'pbpaste', args: [] },
        ]
    }

    return [
        { command: 'wl-paste', args: ['--type', 'text'] },
        { command: 'xclip', args: ['-selection', 'clipboard', '-out'] },
        { command: 'xsel', args: ['--clipboard', '--output'] },
    ]
}

function runClipboardCommand (command: string, args: string[]): Promise<string> {
    return new Promise((resolve, reject) => {
        execFile(command, args, {
            encoding: 'utf8',
            maxBuffer: READ_CLIPBOARD_MAX_BUFFER,
            timeout: READ_CLIPBOARD_TIMEOUT_MS,
            windowsHide: true,
        }, (error, stdout) => {
            if (error) {
                reject(error)
                return
            }

            resolve(stdout)
        })
    })
}

export function getCtrlCExitState (
    previousCtrlCAt: number|null,
    now: number,
): { shouldExit: boolean, lastCtrlCAt: number|null } {
    if (previousCtrlCAt !== null && now - previousCtrlCAt <= CTRL_C_EXIT_WINDOW_MS) {
        return { shouldExit: true, lastCtrlCAt: null }
    }

    return { shouldExit: false, lastCtrlCAt: now }
}

export type HardExitNowOptions = {
    inputName: string
    logger?: Pick<RunLogger, 'log'|'close'>|null
    restoreTerminal?: () => void
    exit?: () => void
    processExit?: (code: number) => never|void
    scheduleProcessExit?: (exitProcess: () => void) => void
}

export function hardExitNow ({
    inputName,
    logger,
    restoreTerminal,
    exit,
    processExit = code => process.exit(code),
    scheduleProcessExit = exitProcess => {
        setTimeout(exitProcess, HARD_EXIT_TUI_TEARDOWN_DELAY_MS)
    },
}: HardExitNowOptions): void {
    try {
        exit?.()
    } catch {
        // Ink teardown is best-effort before process termination.
    }

    try {
        restoreTerminal?.()
    } catch {
        // Keep termination independent from terminal cleanup failures.
    }

    try {
        logger?.log('hard_exit_requested', { input: inputName, code: CTRL_C_HARD_EXIT_CODE })
    } catch {
        // Hard exit must not wait on or be blocked by logging.
    }

    try {
        logger?.close()
    } catch {
        // Ignore fsync/close failures; Ctrl+C hard exit should remain immediate.
    }

    scheduleProcessExit(() => {
        processExit(CTRL_C_HARD_EXIT_CODE)
    })
}

function isModalStage (stage: AppStage): boolean {
    return stage === 'start-menu'
        || stage === 'setup-wizard'
        || stage === 'settings'
        || stage === 'settings-edit'
        || stage === 'select-task'
        || stage === 'select-glossary'
        || stage === 'confirm-config'
        || stage === 'task-complete'
}

export function getHorizontalSettingsFocusArea (
    current: SettingsFocusArea,
    step: number,
): SettingsFocusArea {
    if (current === 'buttons') {
        return current
    }

    return step < 0 ? 'categories' : 'fields'
}

export function getHelpText (stage: AppStage, text: TuiText = getTuiText()): string {
    switch (stage) {
        case 'start-menu':
            return text.help.startMenu
        case 'setup-wizard':
            return text.help.setupWizard
        case 'settings':
            return text.help.settings
        case 'settings-edit':
            return text.help.settingsEdit
        case 'select-task':
        case 'select-glossary':
            return text.help.taskSelection
        case 'confirm-config':
            return text.help.confirmConfig
        case 'task-complete':
            return text.help.taskComplete
        case 'select-directory':
            return text.help.selectDirectory
        // Currently dead code: Ask mode is disabled; kept for possible future reuse.
        case 'ready':
            return text.help.ready
        case 'checking-setup':
        case 'initializing':
        case 'preflighting':
        case 'extracting':
        case 'exporting':
        // Currently dead code: Ask mode is disabled; kept for possible future reuse.
        case 'running':
            return text.help.working
    }
}

export function formatTaskConfigMismatchWarning (task: ProjectTask, differences: TaskConfigDifference[], text: TuiText = getTuiText()): string {
    return text.configMismatch.warning(task.task_code, formatTaskConfigDifferences(differences, 6, text))
}

export function formatTaskConfigDifferences (differences: TaskConfigDifference[], maxItems = 6, text: TuiText = getTuiText()): string {
    const visible = differences.slice(0, Math.max(0, maxItems))
    const lines = visible.map(difference => text.configMismatch.difference(String(difference.field), formatConfigValue(difference.currentValue), formatConfigValue(difference.taskValue)))
    const remaining = differences.length - visible.length

    return [
        ...lines,
        ...(remaining > 0 ? [text.configMismatch.remainingDifferences(remaining)] : []),
    ].join(' ')
}

function formatConfigValue (value: unknown): string {
    return value === null ? 'null' : JSON.stringify(value)
}

export function getDisplayWidth (value: string): number {
    return stringWidth(value)
}

function getButtonLabelWidth (label: string): number {
    return getDisplayWidth(`[ ${label} ]`)
}

function getButtonRowBounds (startColumn: number, availableWidth: number, labels: string[]): ButtonRowBounds {
    const safeAvailableWidth = Math.max(1, availableWidth)
    const naturalWidths = labels.map(getButtonLabelWidth)
    const gapWidth = TASK_DIALOG_BUTTON_GAP_COLUMNS * Math.max(0, labels.length - 1)
    const naturalTotalWidth = naturalWidths.reduce((total, width) => total + width, 0) + gapWidth
    const widths = naturalTotalWidth <= safeAvailableWidth
        ? naturalWidths
        : fitButtonWidths(naturalWidths, Math.max(labels.length, safeAvailableWidth - gapWidth))
    const bounds: ButtonRowBounds = []
    let currentStartColumn = startColumn

    for (const width of widths) {
        const safeWidth = Math.max(1, width)
        bounds.push({
            startColumn: currentStartColumn,
            endColumn: currentStartColumn + safeWidth - 1,
        })
        currentStartColumn += safeWidth + TASK_DIALOG_BUTTON_GAP_COLUMNS
    }

    return bounds
}

function getRightAlignedButtonRowWidths (availableWidth: number, labels: string[]): number[] {
    const safeAvailableWidth = Math.max(1, availableWidth)
    const naturalWidths = labels.map(getButtonLabelWidth)
    const gapWidth = TASK_DIALOG_BUTTON_GAP_COLUMNS * Math.max(0, labels.length - 1)
    const naturalTotalWidth = naturalWidths.reduce((total, width) => total + width, 0) + gapWidth

    return naturalTotalWidth <= safeAvailableWidth
        ? naturalWidths
        : fitButtonWidths(naturalWidths, Math.max(labels.length, safeAvailableWidth - gapWidth))
}

function fitButtonWidths (widths: number[], availableButtonWidth: number): number[] {
    const fittedWidths = widths.map(width => Math.max(1, width))
    let excess = fittedWidths.reduce((total, width) => total + width, 0) - Math.max(fittedWidths.length, availableButtonWidth)

    while (excess > 0) {
        const widestWidth = Math.max(...fittedWidths)
        const widestIndex = fittedWidths.findIndex(width => width === widestWidth)

        if (widestIndex === -1 || fittedWidths[widestIndex] <= 1) {
            break
        }

        fittedWidths[widestIndex] -= 1
        excess -= 1
    }

    return fittedWidths
}

function getStartMenuButtonWidth (text: TuiText): number {
    return Math.max(
        ...START_MENU_CHOICES.map(choice => getButtonLabelWidth(text.startMenu.options[choice])),
    )
}

function padEndDisplayWidth (value: string, width: number): string {
    return value + ' '.repeat(Math.max(0, width - getDisplayWidth(value)))
}

function centerDisplayWidth (value: string, width: number): string {
    const visibleWidth = getDisplayWidth(value)
    const padding = Math.max(0, width - visibleWidth)
    const leftPadding = Math.floor(padding / 2)
    const rightPadding = padding - leftPadding

    return `${' '.repeat(leftPadding)}${value}${' '.repeat(rightPadding)}`
}

function ConfigMismatchDialog ({
    differences,
    height,
    hoveredChoice,
    layout,
    task,
    text,
    terminalColumns,
}: {
    differences: TaskConfigDifference[]
    height: number
    hoveredChoice: ConfigMismatchChoice|null
    layout: ConfigMismatchLayout
    task: ProjectTask
    text: TuiText
    terminalColumns: number
}): React.ReactElement {
    void terminalColumns
    const dialogWidth = Math.max(1, layout.endColumn - layout.startColumn + 1)
    const contentWidth = Math.max(1, dialogWidth - 4)
    const visibleDifferences = getVisibleConfigMismatchDifferences(differences, height)
    const hiddenDifferenceCount = Math.max(0, differences.length - visibleDifferences.length)

    return (
        <Box height={height} justifyContent="center" alignItems="center" width="100%">
            <Box
                borderStyle="round"
                borderColor="yellow"
                flexDirection="column"
                paddingX={1}
                paddingY={CONFIG_MISMATCH_PADDING_Y}
                width={dialogWidth}
            >
                <Text color="yellow" bold wrap="truncate-end">
                    {text.configMismatch.title(task.task_code)}
                </Text>
                <Text wrap="truncate-end">
                    {text.configMismatch.message}
                </Text>
                <Box flexDirection="column" marginTop={CONFIG_MISMATCH_DIFFERENCE_MARGIN_TOP_ROWS}>
                    {visibleDifferences.map(difference => (
                        <Text key={String(difference.field)} wrap="truncate-end">
                            {truncateTerminalText(text.configMismatch.difference(String(difference.field), formatConfigValue(difference.currentValue), formatConfigValue(difference.taskValue)), contentWidth)}
                        </Text>
                    ))}
                    {hiddenDifferenceCount > 0 ? (
                        <Text dimColor wrap="truncate-end">
                            {text.configMismatch.moreDifferences(hiddenDifferenceCount)}
                        </Text>
                    ) : null}
                </Box>
                <Box marginTop={CONFIG_MISMATCH_BUTTON_MARGIN_TOP_ROWS}>
                    {renderButtonLabel(text.configMismatch.useTaskConfig, hoveredChoice !== 'back')}
                    <Text> </Text>
                    {renderButtonLabel(text.configMismatch.back, hoveredChoice === 'back')}
                </Box>
                <Text dimColor wrap="truncate-end">{text.configMismatch.help}</Text>
            </Box>
        </Box>
    )
}

function TaskCompleteDialog ({
    height,
    hoveredTarget,
    layout,
    mode,
    text,
}: {
    height: number
    hoveredTarget: TaskCompleteMouseTarget|null
    layout: TaskCompleteLayout
    mode: AppMode
    text: TuiText
}): React.ReactElement {
    const dialogWidth = Math.max(1, layout.endColumn - layout.startColumn + 1)
    const contentWidth = Math.max(1, dialogWidth - 4)

    return (
        <Box height={height} justifyContent="center" alignItems="center" width="100%">
            <Box
                borderStyle="round"
                borderColor="green"
                flexDirection="column"
                paddingX={1}
                paddingY={TASK_COMPLETE_PADDING_Y}
                width={dialogWidth}
            >
                <Text color="green" bold wrap="truncate-end">
                    {text.taskComplete.title}
                </Text>
                <Text wrap="truncate-end">
                    {truncateTerminalText(mode === 'translation' ? text.taskComplete.translationMessage : text.taskComplete.glossaryMessage, contentWidth)}
                </Text>
                <Text dimColor wrap="truncate-end">
                    {truncateTerminalText(text.taskComplete.nextStep, contentWidth)}
                </Text>
                <Box marginTop={TASK_COMPLETE_BUTTON_MARGIN_TOP_ROWS}>
                    {renderButtonLabel(text.taskComplete.backToMenu, hoveredTarget?.kind === 'button')}
                </Box>
                <Text dimColor wrap="truncate-end">{text.taskComplete.help}</Text>
            </Box>
        </Box>
    )
}

function StartMenu ({
    height,
    hoveredChoice,
    selectedIndex,
    text,
    terminalColumns,
}: {
    height: number
    hoveredChoice: StartMenuChoice|null
    selectedIndex: number
    text: TuiText
    terminalColumns: number
}): React.ReactElement {
    const dialogWidth = getStartMenuDialogWidth(terminalColumns)
    const buttonWidth = Math.min(getStartMenuButtonWidth(text), dialogWidth)

    return (
        <Box height={height} justifyContent="center" alignItems="center" width="100%">
            <Box borderStyle="round" borderColor="cyan" flexDirection="column" paddingX={2} paddingY={START_MENU_PADDING_Y} width={dialogWidth}>
                <Text color="cyan" bold wrap="truncate-end">{text.appTitle}</Text>
                <Text dimColor wrap="truncate-end">{text.startMenu.subtitle}</Text>
                <Box marginTop={START_MENU_OPTION_MARGIN_TOP_ROWS} flexDirection="column" alignItems="center">
                    {START_MENU_CHOICES.map((choice, index) => (
                        <React.Fragment key={choice}>
                            {index > 0 ? <Box height={START_MENU_OPTION_GAP_ROWS} /> : null}
                            {renderMenuButton(text.startMenu.options[choice], selectedIndex === index || hoveredChoice === choice, buttonWidth)}
                        </React.Fragment>
                    ))}
                </Box>
            </Box>
        </Box>
    )
}

function renderMenuButton (label: string, selected: boolean, width: number): React.ReactElement {
    const safeWidth = Math.max(1, width)
    const innerWidth = Math.max(0, safeWidth - 4)
    const visibleLabel = truncateTerminalText(label, innerWidth)
    const paddedLabel = centerDisplayWidth(visibleLabel, innerWidth)
    const buttonText = truncateTerminalText(`[ ${paddedLabel} ]`, safeWidth)

    return (
        <Text color={selected ? 'black' : 'cyan'} backgroundColor={selected ? 'cyan' : undefined} bold={selected}>
            {buttonText}
        </Text>
    )
}

function SettingsPage ({
    config,
    error,
    focusArea,
    height,
    hoveredTarget,
    layout,
    preferences,
    saveMessage,
    selectedButtonIndex,
    selectedCategoryIndex,
    selectedFieldIndex,
    text,
}: {
    config: AgentConfig
    error: string|null
    focusArea: SettingsFocusArea
    height: number
    hoveredTarget: SettingsMouseTarget|null
    layout: SettingsLayout
    preferences: Preferences
    saveMessage: string|null
    selectedButtonIndex: number
    selectedCategoryIndex: number
    selectedFieldIndex: number
    text: TuiText
}): React.ReactElement {
    const dialogWidth = Math.max(1, layout.endColumn - layout.startColumn + 1)
    const category = settingsCategories[selectedCategoryIndex] ?? settingsCategories[0]!
    const contentWidth = Math.max(1, dialogWidth - 4)
    const categoryWidth = Math.max(1, layout.leftEndColumn - layout.leftStartColumn + 1)
    const fieldWidth = Math.max(1, layout.rightEndColumn - layout.rightStartColumn + 1)

    return (
        <Box height={height} justifyContent="center" alignItems="center" width="100%">
            <Box borderStyle="round" borderColor="cyan" flexDirection="column" paddingX={1} paddingY={SETTINGS_DIALOG_PADDING_Y} width={dialogWidth}>
                <Text color="cyan" bold wrap="truncate-end">{text.settings.title}</Text>
                <Text dimColor wrap="truncate-end">{text.settings.subtitle}</Text>
                <Text color={error ? 'red' : undefined} wrap="truncate-end">
                    {error ? truncateTerminalText(error, contentWidth) : ' '}
                </Text>
                <Box flexDirection="row" marginTop={SETTINGS_DIALOG_LIST_MARGIN_TOP_ROWS}>
                    <Box flexDirection="column" width={categoryWidth}>
                        {settingsCategories.map((item, index) => {
                            const selected = selectedCategoryIndex === index || hoveredTarget?.kind === 'category' && hoveredTarget.index === index
                            const focused = focusArea === 'categories' && selectedCategoryIndex === index

                            return (
                                <Text key={item.id} color={selected ? 'cyan' : undefined} bold={selected || focused} wrap="truncate-end">
                                    {focused ? '>' : ' '}{truncateTerminalText(text.settings.categories[item.id as keyof typeof text.settings.categories] ?? item.id, categoryWidth - 2)}
                                </Text>
                            )
                        })}
                    </Box>
                    <Box flexDirection="column" height={maxSettingsFieldCount} marginLeft={2} width={fieldWidth}>
                        {category.fields.map((field, index) => {
                            const selected = selectedFieldIndex === index || hoveredTarget?.kind === 'field' && hoveredTarget.index === index
                            const focused = focusArea === 'fields' && selectedFieldIndex === index
                            const value = formatSettingsDisplayValue(field, config, preferences, text)
                            const label = getSettingsFieldLabel(field, text)
                            const labelWidth = Math.min(24, Math.max(10, Math.floor(fieldWidth * 0.38)))
                            const valueWidth = Math.max(1, fieldWidth - labelWidth - 5)

                            return (
                                <Text key={field.field} color={selected ? 'green' : undefined} bold={selected || focused} wrap="truncate-end">
                                    {focused ? '>' : ' '}
                                    {padEndDisplayWidth(truncateTerminalText(label, labelWidth), labelWidth)}
                                    {'  '}
                                    <Text dimColor={!selected}>{truncateTerminalText(value, valueWidth)}</Text>
                                </Text>
                            )
                        })}
                    </Box>
                </Box>
                <Box marginTop={SETTINGS_DIALOG_BUTTON_MARGIN_TOP_ROWS}>
                    {renderButtonLabel(text.settings.save, focusArea === 'buttons' && selectedButtonIndex === 0 || hoveredTarget?.kind === 'button' && hoveredTarget.button === 'save')}
                    <Text> </Text>
                    {renderButtonLabel(text.settings.back, focusArea === 'buttons' && selectedButtonIndex === 1 || hoveredTarget?.kind === 'button' && hoveredTarget.button === 'back')}
                </Box>
                <Text color={saveMessage === text.settings.saved ? 'green' : saveMessage ? 'red' : undefined} wrap="truncate-end">
                    {saveMessage ? truncateTerminalText(saveMessage, contentWidth) : ' '}
                </Text>
            </Box>
        </Box>
    )
}

function SetupWizardPage ({
    apiTest,
    config,
    error,
    fieldIndex,
    focusArea,
    height,
    hoveredTarget,
    layout,
    preferences,
    saveMessage,
    selectedButton,
    stepIndex,
    text,
}: {
    apiTest: SetupWizardApiTestState
    config: AgentConfig
    error: string|null
    fieldIndex: number
    focusArea: SetupWizardFocusArea
    height: number
    hoveredTarget: SetupWizardMouseTarget|null
    layout: SetupWizardLayout
    preferences: Preferences
    saveMessage: string|null
    selectedButton: SetupWizardButtonFocus
    stepIndex: number
    text: TuiText
}): React.ReactElement {
    const dialogWidth = Math.max(1, layout.endColumn - layout.startColumn + 1)
    const dialogHeight = Math.max(1, layout.endRow - layout.startRow + 1)
    const contentWidth = Math.max(1, dialogWidth - 4)
    const stepWidth = Math.max(1, layout.leftEndColumn - layout.leftStartColumn + 1)
    const fieldWidth = Math.max(1, layout.rightEndColumn - layout.rightStartColumn + 1)
    const bodyRows = Math.max(1, layout.buttonRow - layout.stepStartRow - SETUP_WIZARD_BUTTON_MARGIN_TOP_ROWS)
    const step = SETUP_WIZARD_STEPS[stepIndex] ?? SETUP_WIZARD_STEPS[0]!
    const primaryButtonSelected = focusArea === 'buttons' && selectedButton === 'primary' || hoveredTarget?.kind === 'button' && hoveredTarget.button === layout.primaryButton
    const retestButtonSelected = focusArea === 'buttons' && selectedButton === 'retest' || hoveredTarget?.kind === 'button' && hoveredTarget.button === 'retest'
    const secondaryButtonSelected = focusArea === 'buttons' && selectedButton === 'secondary' || hoveredTarget?.kind === 'button' && hoveredTarget.button === 'skipRecommended'
    const previousButtonSelected = focusArea === 'buttons' && selectedButton === 'previous' || hoveredTarget?.kind === 'button' && hoveredTarget.button === 'previous'
    const buttonSpacerWidth = Math.max(0, layout.previousButtonStartColumn - layout.leftStartColumn)
    const staticLines = getSetupWizardStaticLines(step.id, text, apiTest)
    const staticLineColor = getSetupWizardStaticLineColor(step.id, apiTest)
    const footerColor = getSetupWizardFooterColor(saveMessage, text)

    return (
        <Box height={height} justifyContent="center" alignItems="center" width="100%">
            <Box borderStyle="round" borderColor="cyan" flexDirection="column" height={dialogHeight} paddingX={1} paddingY={SETUP_WIZARD_PADDING_Y} width={dialogWidth}>
                <Text color="cyan" bold wrap="truncate-end">{text.setupWizard.title}</Text>
                <Text dimColor wrap="truncate-end">{text.setupWizard.subtitle}</Text>
                <Text color={error ? 'red' : undefined} wrap="truncate-end">
                    {error ? truncateTerminalText(error, contentWidth) : ' '}
                </Text>
                <Box flexDirection="row" height={bodyRows} marginTop={SETUP_WIZARD_LIST_MARGIN_TOP_ROWS}>
                    <Box flexDirection="column" width={stepWidth}>
                        {SETUP_WIZARD_STEPS.map((item, index) => {
                            const selected = stepIndex === index
                            const marker = index < stepIndex ? '*' : `${index + 1}`

                            return (
                                <Text key={item.id} color={selected ? 'cyan' : undefined} bold={selected} wrap="truncate-end">
                                    {selected ? '>' : ' '}{marker} {truncateTerminalText(text.setupWizard.steps[item.id], stepWidth - 4)}
                                </Text>
                            )
                        })}
                    </Box>
                    <Box flexDirection="column" marginLeft={2} width={fieldWidth}>
                        <Text color="green" bold wrap="truncate-end">{text.setupWizard.steps[step.id]}</Text>
                        <Text dimColor wrap="truncate-end">{truncateTerminalText(text.setupWizard.stepDescriptions[step.id], fieldWidth)}</Text>
                        <Box flexDirection="column" marginTop={1}>
                            {step.fields.length === 0
                                ? staticLines.map((line, index) => (
                                    <Text key={`${line}-${index}`} color={index === 0 ? staticLineColor : undefined} wrap="truncate-end">
                                        {truncateTerminalText(line, fieldWidth)}
                                    </Text>
                                ))
                                : step.fields.map((field, index) => {
                                    const definition = settingsFieldDefinitions.get(field)

                                    if (!definition) {
                                        return null
                                    }

                                    const selected = fieldIndex === index || hoveredTarget?.kind === 'field' && hoveredTarget.index === index
                                    const focused = focusArea === 'fields' && fieldIndex === index
                                    const label = getSettingsFieldLabel(definition, text)
                                    const value = formatSettingsDisplayValue(definition, config, preferences, text)
                                    const labelWidth = Math.min(24, Math.max(10, Math.floor(fieldWidth * 0.38)))
                                    const valueWidth = Math.max(1, fieldWidth - labelWidth - 5)

                                    return (
                                        <Text key={field} color={selected ? 'green' : undefined} bold={selected || focused} wrap="truncate-end">
                                            {focused ? '>' : ' '}
                                            {padEndDisplayWidth(truncateTerminalText(label, labelWidth), labelWidth)}
                                            {'  '}
                                            <Text dimColor={!selected}>{truncateTerminalText(value, valueWidth)}</Text>
                                        </Text>
                                    )
                                })}
                        </Box>
                    </Box>
                </Box>
                <Box marginTop={SETUP_WIZARD_BUTTON_MARGIN_TOP_ROWS}>
                    {buttonSpacerWidth > 0 ? <Box width={buttonSpacerWidth} /> : null}
                    {layout.previousEnabled ? renderButtonLabel(text.setupWizard.previous, previousButtonSelected) : <Text dimColor>{`[ ${text.setupWizard.previous} ]`}</Text>}
                    <Text> </Text>
                    {layout.retestButtonStartColumn !== null ? (
                        <>
                            {renderButtonLabel(text.setupWizard.retest, retestButtonSelected)}
                            <Text> </Text>
                        </>
                    ) : null}
                    {layout.secondaryButtonStartColumn !== null ? (
                        <>
                            {renderButtonLabel(text.setupWizard.skipRecommended, secondaryButtonSelected)}
                            <Text> </Text>
                        </>
                    ) : null}
                    {renderButtonLabel(getSetupWizardPrimaryButtonLabel(layout.primaryButton, text), primaryButtonSelected)}
                </Box>
                <Text color={footerColor} wrap="truncate-end">
                    {saveMessage ? truncateTerminalText(saveMessage, contentWidth) : text.setupWizard.help}
                </Text>
            </Box>
        </Box>
    )
}

function SettingsEditDialog ({
    config,
    edit,
    error,
    height,
    hoveredTarget,
    layout,
    preferences,
    text,
    terminalColumns,
}: {
    config: AgentConfig
    edit: SettingsEditState
    error: string|null
    height: number
    hoveredTarget: SettingsEditMouseTarget|null
    layout: SettingsEditLayout
    preferences: Preferences
    text: TuiText
    terminalColumns: number
}): React.ReactElement|null {
    void terminalColumns
    const definition = settingsFieldDefinitions.get(edit.field)

    if (!definition) {
        return null
    }

    const dialogWidth = Math.max(1, layout.endColumn - layout.startColumn + 1)
    const contentWidth = Math.max(1, dialogWidth - 4)
    const options = getSettingsEditOptions(definition.kind)
    const inputRows = getSettingsEditInputRows(options.length)
    const inputFocused = options.length === 0 && hoveredTarget === null

    return (
        <Box height={height} justifyContent="center" alignItems="center" width="100%">
            <Box borderStyle="double" borderColor="yellow" flexDirection="column" paddingX={1} paddingY={SETTINGS_DIALOG_PADDING_Y} width={dialogWidth}>
                <Text color="yellow" bold wrap="truncate-end">{getSettingsFieldLabel(definition, text)}</Text>
                <Text dimColor wrap="truncate-end">{truncateTerminalText(getSettingsFieldDescription(definition, text), contentWidth)}</Text>
                <Text color={error ? 'red' : undefined} wrap="truncate-end">
                    {error ? truncateTerminalText(error, contentWidth) : ' '}
                </Text>
                <Text wrap="truncate-end">{text.settings.current}: {formatSettingsDisplayValue(definition, config, preferences, text)}</Text>
                <Box flexDirection="column" height={inputRows} marginTop={SETTINGS_EDIT_DIALOG_INPUT_MARGIN_TOP_ROWS}>
                    {options.length > 0
                        ? options.map((option, index) => {
                            const selected = edit.selectedOptionIndex === index || hoveredTarget?.kind === 'option' && hoveredTarget.index === index

                            return (
                                <Text key={option} color={selected ? 'green' : undefined} bold={selected} wrap="truncate-end">
                                    {selected ? '>' : ' '} {option}
                                </Text>
                            )
                        })
                        : (
                            <SettingsEditInputLine
                                contentWidth={contentWidth}
                                input={edit.input}
                                isFocused={inputFocused}
                                isSecret={definition.kind === 'secret'}
                                text={text}
                            />
                        )}
                </Box>
                <Box marginTop={SETTINGS_EDIT_DIALOG_BUTTON_MARGIN_TOP_ROWS}>
                    {renderButtonLabel(text.taskDialog.ok, hoveredTarget?.kind === 'confirm')}
                    <Text> </Text>
                    {renderButtonLabel(text.taskDialog.cancel, hoveredTarget?.kind === 'cancel')}
                </Box>
            </Box>
        </Box>
    )
}

function SettingsEditInputLine ({
    contentWidth,
    input,
    isFocused,
    isSecret,
    text,
}: {
    contentWidth: number
    input: string
    isFocused: boolean
    isSecret: boolean
    text: TuiText
}): React.ReactElement {
    const emptySecretValue = isSecret && input.length === 0
    const displayValue = emptySecretValue ? text.settings.emptySecret : input
    const cursor = isFocused && !emptySecretValue ? '|' : ''
    const inputText = truncateTerminalText(`${text.settings.value}: ${displayValue}${cursor}`, contentWidth)

    if (isFocused) {
        return (
            <Text backgroundColor="green" color="black" bold wrap="truncate-end">
                {padEndDisplayWidth(inputText, Math.max(0, contentWidth))}
            </Text>
        )
    }

    return (
        <Text wrap="truncate-end">
            {text.settings.value}: <Text color="green">{displayValue}</Text>
        </Text>
    )
}

function TaskSelectionDialog ({
    hoveredTarget,
    height,
    mode,
    newTaskLabel,
    promptText,
    runDirectory,
    selectedIndex,
    text,
    title,
    tasks,
    terminalColumns,
}: {
    hoveredTarget: TaskSelectionMouseTarget|null
    height: number
    mode: AppMode
    newTaskLabel?: string
    promptText?: string
    runDirectory: string
    selectedIndex: number
    text: TuiText
    title?: string
    tasks: ProjectTask[]
    terminalColumns: number
}): React.ReactElement {
    const visibleTasks = getVisibleTaskChoices(tasks)
    const choiceCount = getTaskSelectionChoiceCount(visibleTasks)
    const dialogWidth = getTaskSelectionDialogWidth(terminalColumns)
    const contentWidth = Math.max(1, dialogWidth - 4)
    const confirmSelected = hoveredTarget?.kind === 'confirm'
    const cancelSelected = hoveredTarget?.kind === 'cancel'
    const changeDirectorySelected = hoveredTarget?.kind === 'change-directory'

    return (
        <Box height={height} justifyContent="center" alignItems="center" width="100%">
            <Box
                borderStyle="round"
                borderColor="cyan"
                flexDirection="column"
                paddingX={1}
                paddingY={TASK_DIALOG_PADDING_Y}
                width={dialogWidth}
            >
                <Text color="cyan" bold wrap="truncate-end">
                    {title ?? (mode === 'translation' ? text.taskDialog.translationTitle : text.taskDialog.glossaryTitle)}
                </Text>
                <Text dimColor wrap="truncate-end">
                    {truncateMiddle(runDirectory, contentWidth)}
                </Text>
                <Text wrap="truncate-end">
                    {promptText ?? (visibleTasks.length === 0 ? text.taskDialog.emptyPrompt : text.taskDialog.defaultPrompt)}
                </Text>
                <Box flexDirection="column" marginTop={TASK_DIALOG_OPTION_MARGIN_TOP_ROWS}>
                    {visibleTasks.map((task, index) => (
                        <TaskSelectionOption
                            key={task.task_code}
                            contentWidth={contentWidth}
                            index={index}
                            isSelected={selectedIndex === index || isHoveredTaskChoice(hoveredTarget, index)}
                            label={formatTaskCardLabel(task, contentWidth - 8, text)}
                            statusColor={getTaskStatusColor(task)}
                            total={choiceCount}
                        />
                    ))}
                    <TaskSelectionOption
                        contentWidth={contentWidth}
                        index={visibleTasks.length}
                        isSelected={selectedIndex === visibleTasks.length || isHoveredTaskChoice(hoveredTarget, visibleTasks.length)}
                        label={newTaskLabel ?? text.taskDialog.newTask}
                        statusColor="cyan"
                        total={choiceCount}
                    />
                </Box>
                <Box marginTop={TASK_DIALOG_BUTTON_MARGIN_TOP_ROWS}>
                    {renderButtonLabel(text.taskDialog.ok, confirmSelected)}
                    <Text> </Text>
                    {renderButtonLabel(text.taskDialog.cancel, cancelSelected)}
                    <Text> </Text>
                    {renderButtonLabel(text.taskDialog.changeDirectory, changeDirectorySelected)}
                </Box>
                <Text dimColor wrap="truncate-end">{text.taskDialog.help}</Text>
            </Box>
        </Box>
    )
}

function isHoveredTaskChoice (target: TaskSelectionMouseTarget|null, index: number): boolean {
    return target?.kind === 'choice' && target.index === index
}

function TaskSelectionOption ({
    contentWidth,
    index,
    isSelected,
    label,
    statusColor,
    total,
}: {
    contentWidth: number
    index: number
    isSelected: boolean
    label: string
    statusColor: string
    total: number
}): React.ReactElement {
    const prefix = `${isSelected ? '>' : ' '} ${index + 1}/${total} `
    const safeLabelWidth = Math.max(1, contentWidth - getDisplayWidth(prefix))

    return (
        <Text color={isSelected ? 'green' : statusColor} bold={isSelected} wrap="truncate-end">
            {prefix}{truncateTerminalText(label, safeLabelWidth)}
        </Text>
    )
}

function truncateMiddle (value: string, maxWidth: number): string {
    return cliTruncate(value, Math.max(0, maxWidth), {
        position: 'middle',
        truncationCharacter: '...',
    })
}

function truncateTerminalText (value: string, maxWidth: number): string {
    return cliTruncate(value, Math.max(0, maxWidth), {
        position: 'end',
        truncationCharacter: '...',
    })
}

function PromptLine ({ stage, input, mode, text }: { stage: AppStage, input: string, mode: AppMode, text: TuiText }): React.ReactElement {
    if (stage === 'checking-setup') {
        return <Text color="yellow" wrap="truncate-end">{text.prompt.checkingSetup}</Text>
    }

    if (stage === 'initializing') {
        return <Text color="yellow" wrap="truncate-end">{text.prompt.initializing}</Text>
    }

    if (stage === 'preflighting') {
        return <Text color="yellow" wrap="truncate-end">{mode === 'translation' ? text.prompt.translationPreflight : text.prompt.glossaryPreflight}</Text>
    }

    if (stage === 'extracting') {
        return <Text color="yellow" wrap="truncate-end">{mode === 'translation' ? text.prompt.translationExtracting : text.prompt.glossaryExtracting}</Text>
    }

    if (stage === 'exporting') {
        return <Text color="yellow" wrap="truncate-end">{text.prompt.exporting}</Text>
    }

    if (stage === 'running') {
        return <Text color="yellow" wrap="truncate-end">{text.prompt.running}</Text>
    }

    if (stage === 'start-menu') {
        return <Text color="yellow" wrap="truncate-end">{text.startMenu.prompt}</Text>
    }

    if (stage === 'settings') {
        return <Text color="yellow" wrap="truncate-end">{text.prompt.settings}</Text>
    }

    if (stage === 'setup-wizard') {
        return <Text color="yellow" wrap="truncate-end">{text.prompt.setupWizard}</Text>
    }

    if (stage === 'settings-edit') {
        return <Text color="yellow" wrap="truncate-end">{text.prompt.settingsEdit}</Text>
    }

    if (stage === 'select-task') {
        return <Text color="yellow" wrap="truncate-end">{text.prompt.selectingTask}</Text>
    }

    if (stage === 'select-glossary') {
        return <Text color="yellow" wrap="truncate-end">{text.prompt.selectingGlossary}</Text>
    }

    if (stage === 'confirm-config') {
        return <Text color="yellow" wrap="truncate-end">{text.prompt.confirmConfig}</Text>
    }

    if (stage === 'task-complete') {
        return <Text color="green" wrap="truncate-end">{text.prompt.taskComplete}</Text>
    }

    if (stage === 'select-directory') {
        return (
            <Text wrap="truncate-end">
                {text.prompt.directory(defaultDirectory)} <Text color="green">{input}</Text>
            </Text>
        )
    }

    // Currently dead code: Ask mode is disabled; kept for possible future reuse.
    return (
        <Text wrap="truncate-end">
            {text.prompt.ask}: <Text color="green">{input}</Text>
        </Text>
    )
}

async function resolveRunDirectory (directoryInput: string, text: TuiText = getTuiText()): Promise<string> {
    const requestedDirectory = directoryInput.trim() || defaultDirectory
    const resolvedDirectory = path.resolve(defaultDirectory, requestedDirectory)
    const directoryStat = await stat(resolvedDirectory)

    if (!directoryStat.isDirectory()) {
        throw new Error(text.message.notDirectory(resolvedDirectory))
    }

    return realpath(resolvedDirectory)
}
