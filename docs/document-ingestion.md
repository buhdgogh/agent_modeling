# 文档上传与解析

上传入口在 `app.py`，解析路由在 `agent_modeling/documents/parser.py`。
解析在用户提交任务时执行；普通界面刷新不会主动运行大型解析模型。

## 格式路由

| 输入 | 正文与结构 | 表格 |
| --- | --- | --- |
| TXT | Python 解码、段落及常见中文章/节标题识别 | 不推测纯文本中的二维表格 |
| MD / Markdown | Python 标题、段落、围栏代码和管道表格解析 | 保留原始管道表格 |
| JSON | Python JSON 校验，保留原始文本 | 不将任意 JSON 推测成表格 |
| HTML / HTM | Python HTMLParser，保留标题与正文顺序 | 保留表格 HTML，包括合并单元格属性 |
| PDF | MinerU 4.x SDK，显式处理全部页面 | MinerU 表格结构和 HTML |
| DOCX | MinerU 4.x SDK | MinerU 表格结构和 HTML |
| DOC | MinerU 4.x SDK | MinerU 表格结构和 HTML |

CSV 仍是现有钻孔/地层配置数据，不自动当作报告正文。图件和空间文件保持原有处理入口。

## 安装和配置

在运行 Streamlit 的 Python 环境安装额外依赖：

```powershell
python -m pip install -r requirements/documents.txt
```

MinerU 适配器按当前官方 4.x 的 `mineru.parser.parse`、`ParseResult.save`、
`RenderFormat.CONTENT_LIST` 接口编写，不兼容旧版 2.x 命令行。
安装 SDK 不代表模型权重和加速运行环境已经准备好；按所选档位配置对应的 MinerU
运行时和模型。

`api_key.env` 可配置：

```dotenv
DOCUMENT_MINERU_TIER=standard
DOCUMENT_MINERU_API_URL=
MINERU_API_KEY=
DOCUMENT_PARSER_CACHE_REVISION=1
```

- `DOCUMENT_MINERU_TIER`：`flash`、`basic`、`standard` 或 `advanced`。
- `DOCUMENT_MINERU_API_URL`：留空调用本地 SDK；填写自建 MinerU V1 服务根地址时，
  SDK 将文件提交到该服务。不要填写到 `/v1` 子路径。
- `MINERU_API_KEY`：可选服务密钥，不进入缓存配置和文档结果。
- `DOCUMENT_PARSER_CACHE_REVISION`：远端模型或解析策略变化时修改，强制生成新缓存。

缺少依赖、服务失败、空内容或不完整解析会报错，不将空结果当作成功继续抽取。
Word/PDF 的正文、结构、表格只调用 MinerU，不再启动其他表格解析器或格式转换进程。

## 统一结果与来源

`parse_document(path)` 返回普通 Python 字典，包含：

- `document_id`、原文件名和内容 SHA256。
- `blocks`：标题、段落、表格、图像等有序内容块。
- `block_id`、`section_path`、解析器来源和原文定位信息。
- `llm_text`：带内容块 ID 的模型可读视图。
- `warnings`：解析警告。

PDF 使用从零开始的 `page_index`，有可用坐标时统一为左上原点、0～1 归一化 bbox。
面向模型显示的 `page` 是从1开始。TXT/MD 保留原始行号；HTML 保留源码行号；
DOCX/DOC 不把解析器的逻辑分页伪装成真实页码，而保留块及解析器定位信息。

会话文档缓存保存在 `runtime/sessions/<会话ID>/documents/<缓存键>/`，避免在多层任务输入目录下
继续拼接长路径，清理会话时也会清理这些缓存。独立短路径文档仍使用文件旁边的
`parsed_documents/<缓存键>/`；深路径文档回退到较短的 `runtime/document_cache/<目录标识>/`。
显式传入 `cache_dir` 时使用调用方指定目录。缓存通过短名称临时文件原子写入，失败时清理临时文件。
其中 `document.json` 是统一结果；`mineru/` 保存 MinerU 原始结构、Markdown、图片。
缓存键包含文件内容、名称、扩展名、解析配置、SDK 版本和缓存修订号。
只有完成解析才原子写入成功缓存；重新提交相同文档可复用缓存。

## 表格处理

Word/PDF 的表格直接使用 MinerU 输出，保留表格 HTML、表题、脚注及可用来源位置。
不再合并或替换另一套解析器的表格，也不需要安装 LibreOffice。
实际表格恢复质量取决于 MinerU 运行时及输入文档，当前没有额外的自动跨页表格拼接。

## 交给文本 Agent

前端把 `llm_text` 提供给主控，另将 `parsed_documents` 作为运行数据传入。
主控再把统一文档交给文本 Agent。报告工具的搜索和读取结果会返回内容块 ID、
源文件及来源位置，避免只保留纯文本后丢失证据定位。

结构化文档按内容块构建抽取单元，目标约6000字符；标题路径和来源 ID 会重复携带。
长正文可拆分，表格保持完整；这里的6000是打包目标，不是完整表格的硬上限。
特别大的表格仍可能超过所选模型上下文，需要另行提供表格分页策略或人工拆分。
直接调用 `TextExtractionAgent.run(text=...)` 且不传结构化文档时，仍使用原有字符切分方式。

图片及图表资产被保存，但文档解析不等于完成地质图义理解；它们仍需视觉分析流程。
当前结构化抽取 Schema 尚未要求每个结果字段返回证据引用，因此来源可通过输入块和
工具观察追溯，但不应声称已实现字段级证据验证。

## 验证范围

```powershell
python -m unittest discover -s tests -v
```

原生解析使用真实文件测试；MinerU 使用 SDK 接口替身验证调用契约、
页码坐标转换、表格保留、缓存与失败处理。文本工具循环使用脚本模型测试。
这些测试不证明 OCR 或复杂表格的实际恢复效果，部署后仍需用真实报告联调。

官方接口依据：

- [MinerU Python SDK](https://opendatalab.github.io/MinerU/usage/sdk_api/)
- [MinerU 输出协议](https://opendatalab.github.io/MinerU/reference/output_files/)
