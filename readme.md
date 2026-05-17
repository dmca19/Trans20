# Trans20

Trans20 is an agent-driven client for glossary extraction and ManualTrans JSON translation.

Agents can use a variety of matching, research, and submission tools to precisely retrieve existing results, review the full source text, and achieve better translation quality.

------

![Trans20 screenshot](https://i.ibb.co/BHYcJ57M/photo-2026-05-16-17-12-21.jpg)

Demo: [https://trans.dmca19.cc/](https://trans.dmca19.cc/)

## 特性

Trans20 是一个由 agent 驱动的术语提取和 ManualTrans JSON 翻译工具。

agent 可以使用多种匹配、调查和提交工具，检索已有结果中需要的部分，并总览全部原文，从而获得更好的翻译效果。

- 支持从 ManualTrans JSON 中抽取术语、证据和译名规则。
- agent 可以使用多种工具主动获取需要的数据。
- 多个 agent 可并行处理批次，并通过共享术语表或翻译状态协作。
- 术语表任务会经过抽取、审核和持久化；翻译任务可复用同一 ManualTransFile 的已完成术语表。




## 使用方法

```sh
pnpm install
cp config.example.json config.json
pnpm start
```

## 配置说明

复制 `config.example.json` 为 `config.json` 后，至少需要配置模型访问参数：

- `url`：OpenAI 兼容 API 的 base URL。
- `key`：API key。
- `model`：要使用的模型名。

这些参数也可以通过环境变量提供：`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`。

将源文件放在 `input/ManualTransFile.json`，或在 `config.json` 中设置 `manualTransFile` 指向其他项目内路径。其他并发、批次大小、日志和质量检查选项可以按需要继续调整。

## 输入格式

`manualTransFile` 必须指向 ManualTrans JSON key-value 翻译表，顶层是源文到当前译文或占位内容的映射：

```json
{
  "イベントを開始します。": "イベントを開始します。"
}
```

术语表模式：

```sh
pnpm start
# 选择 Start Glossary
```

翻译模式：

```sh
pnpm start
# 选择 Start Translation
```

每次运行都会在 `output/<task_code>/` 下创建一个独立的任务目录。常见输出包括：

- `task.json`：任务元数据、配置快照和进度。
- `glossary.json`：术语表状态，术语表任务会写入。
- `translation-state.json`：翻译状态，翻译任务会写入。
- `ManualTransFile_translated.json`：翻译后的 ManualTrans JSON，翻译任务完成导出时写入。
- `logs/`：任务相关日志。
- `snapshots/`：agent 生命周期快照。

翻译任务可以选择同一 ManualTransFile 哈希下已完成的术语表任务，也可以不使用术语表直接继续。匹配基于配置的 ManualTransFile 的 SHA-256 哈希，因此即使等价源文件路径发生变化，也不会影响任务匹配。



## agent 工具能力

agent 会在不同阶段获得对应的工具能力：

- 总览工具：检查 ManualTrans 格式，统计 key 数量和样例，分析语言混合、占位符、路径、符号比例、高频词和重复短语等整体风险。
- 通用工具：按范围读取源文 key，搜索关键词或正则，查看匹配上下文，查询已有术语、术语条目和证据上下文。
- 术语表抽取：创建或更新术语，补充别名、类型、译名规则、证据和上下文，并提交批次抽取结果。
- 术语表审核：审核新增术语和候选证据，处理重复、冲突、废弃、合并和修订，并提交审核结论。
- 翻译执行：获取待翻译批次，查询已批准术语和证据上下文，读取前后原文、已提交译文和翻译记忆，并提交批次译文。
