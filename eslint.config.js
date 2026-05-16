import js from '@eslint/js'
import stylistic from '@stylistic/eslint-plugin'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import tseslint from 'typescript-eslint'

const maxCompactUnionMembers = 5
const maxCompactUnionLength = 100

function getTypeParameterNodes (node) {
    return node.typeArguments?.params ?? node.typeParameters?.params ?? []
}

function isSimpleUnionMember (node) {
    switch (node.type) {
        case 'TSAnyKeyword':
        case 'TSBigIntKeyword':
        case 'TSBooleanKeyword':
        case 'TSNeverKeyword':
        case 'TSNullKeyword':
        case 'TSNumberKeyword':
        case 'TSObjectKeyword':
        case 'TSStringKeyword':
        case 'TSSymbolKeyword':
        case 'TSThisType':
        case 'TSUndefinedKeyword':
        case 'TSUnknownKeyword':
        case 'TSVoidKeyword':
        case 'TSLiteralType':
        case 'TSTypeQuery':
            return true
        case 'TSArrayType':
            return isSimpleUnionMember(node.elementType)
        case 'TSIndexedAccessType':
            return isSimpleUnionMember(node.objectType) && isSimpleUnionMember(node.indexType)
        case 'TSParenthesizedType':
            return isSimpleUnionMember(node.typeAnnotation)
        case 'TSTypeOperator':
            return isSimpleUnionMember(node.typeAnnotation)
        case 'TSTypeReference':
            return getTypeParameterNodes(node).every(isSimpleUnionMember)
        case 'TSUnionType':
            return node.types.every(isSimpleUnionMember)
        default:
            return false
    }
}

function shouldRequireCompactUnion (sourceCode, node) {
    if (node.types.length > maxCompactUnionMembers || !node.types.every(isSimpleUnionMember)) {
        return false
    }

    const compactText = node.types.map(typeNode => sourceCode.getText(typeNode)).join('|')

    return compactText.length <= maxCompactUnionLength
}

const compactSimpleUnionTypesRule = {
    meta: {
        type: 'layout',
        docs: {
            description: 'Require compact TypeScript union type separators for simple unions',
        },
        messages: {
            compactUnion: 'Simple union type separator `|` must be compact, for example `string|null`.',
        },
        schema: [],
    },
    create (context) {
        const sourceCode = context.sourceCode

        return {
            TSUnionType (node) {
                if (!shouldRequireCompactUnion(sourceCode, node)) {
                    return
                }

                const firstToken = sourceCode.getFirstToken(node)

                if (firstToken?.value === '|') {
                    context.report({
                        loc: firstToken.loc,
                        messageId: 'compactUnion',
                        node,
                    })
                }

                for (let index = 1; index < node.types.length; index += 1) {
                    const previousType = node.types[index - 1]
                    const nextType = node.types[index]
                    const token = sourceCode.getTokenAfter(previousType, item => item.value === '|')
                    const previousToken = sourceCode.getLastToken(previousType)
                    const nextToken = sourceCode.getFirstToken(nextType)

                    if (!token || !previousToken) {
                        context.report({
                            loc: token?.loc ?? nextToken?.loc ?? node.loc,
                            messageId: 'compactUnion',
                            node,
                        })
                        continue
                    }

                    if (!nextToken) {
                        continue
                    }

                    const before = sourceCode.text.slice(previousToken.range[1], token.range[0])
                    const after = sourceCode.text.slice(token.range[1], nextToken.range[0])

                    if (before !== '' || after !== '') {
                        context.report({
                            loc: token.loc,
                            messageId: 'compactUnion',
                            node,
                        })
                    }
                }
            },
        }
    },
}

const project = {
    rules: {
        'compact-union-types': compactSimpleUnionTypesRule,
    },
}

const codeStyleRules = {
    '@stylistic/array-bracket-spacing': ['error', 'never'],
    '@stylistic/arrow-spacing': ['error', { before: true, after: true }],
    '@stylistic/brace-style': ['error', '1tbs', { allowSingleLine: true }],
    '@stylistic/comma-dangle': ['error', 'always-multiline'],
    '@stylistic/comma-spacing': ['error', { before: false, after: true }],
    '@stylistic/eol-last': ['error', 'always'],
    '@stylistic/indent': ['error', 4, { SwitchCase: 1 }],
    '@stylistic/jsx-indent-props': ['error', 4],
    '@stylistic/jsx-quotes': ['error', 'prefer-double'],
    '@stylistic/key-spacing': ['error', { beforeColon: false, afterColon: true }],
    '@stylistic/keyword-spacing': ['error', { before: true, after: true }],
    '@stylistic/linebreak-style': ['error', 'unix'],
    '@stylistic/no-extra-semi': 'error',
    '@stylistic/no-mixed-spaces-and-tabs': 'error',
    '@stylistic/no-multi-spaces': 'error',
    '@stylistic/no-multiple-empty-lines': ['error', { max: 1, maxBOF: 0, maxEOF: 0 }],
    '@stylistic/no-tabs': 'error',
    '@stylistic/no-trailing-spaces': 'error',
    '@stylistic/object-curly-spacing': ['error', 'always'],
    '@stylistic/quotes': ['error', 'single', { avoidEscape: true }],
    '@stylistic/semi': ['error', 'never'],
    '@stylistic/space-before-blocks': ['error', 'always'],
    '@stylistic/space-before-function-paren': ['error', {
        anonymous: 'always',
        asyncArrow: 'always',
        named: 'always',
    }],
    '@stylistic/space-in-parens': ['error', 'never'],
    '@stylistic/template-curly-spacing': ['error', 'never'],
    '@stylistic/type-annotation-spacing': 'error',
    '@stylistic/type-generic-spacing': 'error',
    '@stylistic/type-named-tuple-spacing': 'error',
}

export default tseslint.config(
    {
        name: 'project/ignores',
        ignores: [
            '**/node_modules/**',
            'dist/**',
            'build/**',
            'coverage/**',
            'logs/**',
            'output/**',
            'democases/**',
            'input/**',
            'data/models/**',
        ],
    },
    {
        name: 'project/javascript',
        files: ['eslint.config.js', '*.config.js', 'scripts/**/*.mjs'],
        languageOptions: {
            ecmaVersion: 'latest',
            sourceType: 'module',
            globals: globals.node,
        },
        plugins: {
            '@stylistic': stylistic,
        },
        rules: {
            ...js.configs.recommended.rules,
            ...codeStyleRules,
        },
    },
    ...tseslint.configs.recommended,
    {
        name: 'project/typescript',
        files: ['src/**/*.{ts,tsx}', 'tests/**/*.{ts,tsx}'],
        languageOptions: {
            ecmaVersion: 'latest',
            sourceType: 'module',
            globals: globals.node,
        },
        plugins: {
            '@stylistic': stylistic,
            project,
            'react-hooks': reactHooks,
        },
        rules: {
            ...codeStyleRules,
            'project/compact-union-types': 'error',
            '@typescript-eslint/consistent-type-imports': ['error', {
                disallowTypeAnnotations: false,
                fixStyle: 'inline-type-imports',
                prefer: 'type-imports',
            }],
            '@typescript-eslint/no-unused-vars': ['warn', {
                argsIgnorePattern: '^_',
                caughtErrorsIgnorePattern: '^_',
                ignoreRestSiblings: true,
                varsIgnorePattern: '^_',
            }],
            'react-hooks/exhaustive-deps': 'warn',
            'react-hooks/rules-of-hooks': 'error',
        },
    },
)
