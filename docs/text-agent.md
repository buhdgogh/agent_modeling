# 文本 Agent：上下文与工具

文本 Agent 使用自己的模型请求和局部消息列表。它与主控可以使用同一个模型服务，
但不会自动获得主控的历史消息。判断是否共享上下文，应检查 `invoke(messages)`
中实际传入的消息，而不是 API Key、模型名称或 Python 类名。

多次 API 调用本身不构成多智能体。当前文本模块增加了自主选择工具、读取工具结果、
继续决策的局部循环；主控通过条件边调度专业图节点。

## 执行方式

1. 有结构化文档时按内容块组织输入，保留完整表格；仅传纯文本时按6000字符、300字符重叠切分。最多五个单元并发。
2. 每个分块建立独立局部消息列表，并获得本次报告章节目录。
3. 模型按需调用工具；每次工具观察以配对的 `ToolMessage` 加入局部历史。
4. 模型再次读取历史，决定继续查询或停止。默认最多三轮、八次实际工具调用。
5. 使用这些消息及业务抽取 Schema 生成地层和剖面结果，再由程序关联来源证据。

`LLMExtraction`使用`ExtractedStratum`和`ExtractedProfile`，仅向模型提供地层、剖面业务字段。
服务端记录`StratumItem`和`ProfileItem`才包含`entity_id`及严格的`evidence`字典列表。
模型额外生成的证据、ID等元数据不会被采信；由`ground_text_records`依据原文重新生成。
这样既不会因模型将`evidence`写成字符串列表而中断，也不会把模型自报证据当作验证结果。
业务字段类型与置信度范围仍需校验；格式失败的重试会反馈字段位置与错误类型。
厚度中的大于、约、未见底和范围须原样保留，不能抹去限定语后作为精确实测厚度。

各分块可检索同一份本次报告库，但不共享彼此的消息列表。报告库与工具在每次运行时
创建，不在共享 Agent 实例上保存当前报告，避免不同请求串用报告。
子图没有跨轮 Checkpoint，也没有自动长期记忆。
单次工具观察最多8000字符，累计观察预算24000字符；超出时返回缩小查询范围的提示。

## 三个工具

| 工具 | 参数 | 数据范围 |
| --- | --- | --- |
| `search_reports` | `query`, `limit=5` | 本次传入报告，关键词匹配，返回报告名、章节 ID、行号和片段 |
| `read_section` | `section_id`, `start_line=1`, `max_lines=40` | 根据目录或检索结果读取章节；行号是章节内部行号，可用 `next_line` 翻页 |
| `lookup_stratum` | `term` | 显式配置的本地地层词典，按名称或别名匹配 |

上传入口通过[文档解析层](document-ingestion.md)传入统一文档，工具返回来源位置与内容块ID。
也可仅传纯文本（识别 `--- 文件名 ---` 标记），或由调用者通过
`reports={"报告名": "报告正文"}` 显式提供多个报告。
章节识别支持 Markdown 标题、中文章/节标题及常见编号标题；无标题时作为正文处理。
这不是互联网搜索，也不会自动扫描其他会话或项目目录。

## 地层词典

设置 `TEXT_AGENT_DICTIONARY_PATH` 为调用者维护的 UTF-8 JSON 文件路径。
示例仅展示结构，不提供未经核实的地质词条：

```json
[
  {
    "name": "填写已核实的地层名称",
    "aliases": ["填写别名或原始地层代号"],
    "description": "填写词典释义",
    "source": "填写词典出处及版本"
  }
]
```

未配置或词典为空时返回 `dictionary_not_configured`；未命中时返回 `not_found`。
词典用于解释术语，不能作为报告中实测厚度或实体对应关系的替代证据。
配置文件无效时明确报错，不静默使用模型常识伪装词典查询。

## 调用和兼容性

```python
agent = TextExtractionAgent(api_key=api_key, dictionary_path="path/to/strata.json")
result = agent.run(
    text="当前报告内容",
    instruction="重点核实上覆、下伏关系和剖面厚度",
    reports={"补充报告": "第一章 地层\n补充资料正文"},
)
extraction = result["extraction"]
trace = extraction.tool_trace
```

原有 `run(text, instruction="")`、`extraction.strata` 和 `extraction.profiles` 接口保留。
可通过 `documents=[统一文档字典]` 传入来源完整的解析结果。
新增 `extraction.tool_trace` 记录分块、轮次、工具参数、返回观察和调用状态。
主控的文本抽取入口及自动钻孔入口都会透传用户指令。

工具模式要求所选模型支持 Tool Calling。工具决策调用失败会抛出项目统一模型异常，
不静默跳过分块。需要使用原有固定抽取路径时，可显式设置 `enable_tools=False`。
新增工具循环会增加模型调用量；分块较多时成本与延迟都会增加。

## 离线验证

`python -m unittest discover -s tests -v`

离线测试覆盖报告作用域、章节读取、长行检索、词典配置，以及用脚本模型验证
工具观察反馈、连续调用、调用上限和未知工具。核心循环通过 AST 加载以避免测试
依赖外部模型 SDK；测试不代替真实 LangChain/DashScope 联调。
