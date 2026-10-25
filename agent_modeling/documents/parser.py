"""Document parsing with native text readers and MinerU for complex formats."""


SIMPLE_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".json", ".html", ".htm"})
COMPLEX_SUFFIXES = frozenset({".pdf", ".docx", ".doc"})
DOCUMENT_SUFFIXES = SIMPLE_SUFFIXES | COMPLEX_SUFFIXES


from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ParsingOptions:
    mineru_tier: str = "standard"
    mineru_api_url: str = ""

    @classmethod
    def from_env(cls):
        return cls(mineru_tier=os.getenv("DOCUMENT_MINERU_TIER", "standard"),
                   mineru_api_url=os.getenv("DOCUMENT_MINERU_API_URL", ""))


def _base_block(kind, text, backend, **source):
    return {"type": kind, "text": text, "backend": backend, "source": source,
            "section_path": []}


import html
from html.parser import HTMLParser
import json
import re
from agent_modeling.errors import FileValidationError


def decode_text(data):
    # Check UTF-32 first: its little-endian BOM starts with the UTF-16 BOM.
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return data.decode("utf-32"), "utf-32"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = data.decode(encoding)
            if "\x00" in text:
                raise FileValidationError("Unexpected binary content in text document")
            return text, encoding
        except UnicodeDecodeError:
            pass
    raise FileValidationError("Unsupported text encoding")


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if not self.skip and tag in {"p", "div", "br", "tr", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")
        if not self.skip and tag in {"td", "th"}:
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)
        if not self.skip and tag in {"p", "div", "tr", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def plain_text(markup):
    parser = _HTMLText()
    parser.feed(str(markup))
    return "".join(parser.parts).strip()


class _HTMLBlocks(HTMLParser):
    """Parse simple HTML structurally; retain table markup and source line anchors."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.parts, self.headings = [], [], []
        self.start, self.table_depth, self.skip, self.heading_level = 1, 0, 0, None

    def flush(self, kind="paragraph"):
        text = "".join(self.parts).strip()
        if text:
            if kind == "heading":
                self.headings = self.headings[:self.heading_level - 1] + [text]
            block = _base_block(kind, text, "python", start_line=self.start,
                                end_line=self.getpos()[0], locator="source_html")
            block["section_path"] = list(self.headings)
            if kind == "heading":
                block["level"] = self.heading_level
            self.blocks.append(block)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if self.skip:
            return
        if tag == "table":
            if not self.table_depth:
                self.flush()
                self.start = self.getpos()[0]
            self.table_depth += 1
        if self.table_depth:
            self.parts.append(self.get_starttag_text())
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.flush()
            self.heading_level, self.start = int(tag[1]), self.getpos()[0]
        elif tag in {"p", "div", "li"}:
            self.flush()
            self.start = self.getpos()[0]
        elif tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if self.table_depth:
            self.parts.append(f"</{tag}>")
            if tag == "table":
                self.table_depth -= 1
                if not self.table_depth:
                    self.flush("table")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and self.heading_level:
            self.flush("heading")
            self.heading_level = None
        elif tag in {"p", "div", "li"}:
            self.flush()

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(html.escape(data) if self.table_depth else data)


def parse_simple(path):
    """Preserve original lines; Markdown pipe tables stay atomic."""
    text, encoding = decode_text(path.read_bytes())
    if path.suffix.lower() == ".json":
        json.loads(text)  # Validate; retain the original serialization and line positions.
    if path.suffix.lower() in {".html", ".htm"}:
        parser = _HTMLBlocks()
        parser.feed(text)
        parser.close()
        parser.flush("table" if parser.table_depth else "paragraph")
        return parser.blocks, {"encoding": encoding}
    lines, blocks, buffer, start = text.splitlines(), [], [], 1
    headings = []

    def flush(kind="paragraph"):
        nonlocal buffer
        if buffer:
            block = _base_block(kind, "\n".join(buffer), "python",
                                start_line=start, end_line=start + len(buffer) - 1,
                                locator="rendered_text" if path.suffix.lower() in {".html", ".htm"} else "source_text")
            block["section_path"] = list(headings)
            blocks.append(block)
            buffer = []

    index, fence = 0, None
    while index < len(lines):
        line = lines[index]
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if fence or fence_match:
            if not fence:
                flush()
                start, fence = index + 1, fence_match[1]
                buffer.append(line)
            elif fence_match and fence_match[1][0] == fence[0] and len(fence_match[1]) >= len(fence):
                buffer.append(line)
                flush("code")
                fence = None
            else:
                buffer.append(line)
            index += 1
            continue
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        chinese_heading = re.match(r"^第[一二三四五六七八九十百零〇\d]+[章节篇].{0,100}$", line.strip())
        table = (index + 1 < len(lines) and "|" in line
                 and re.match(r"^\s*\|?\s*:?-{3,}", lines[index + 1]))
        if match or chinese_heading:
            flush()
            level, title = (len(match[1]), match[2]) if match else (2 if "节" in line else 1, line.strip())
            headings = headings[:level - 1] + [title]
            block = _base_block("heading", title, "python", start_line=index + 1, end_line=index + 1)
            block.update(level=level, section_path=list(headings))
            blocks.append(block)
        elif table:
            flush()
            table_start, table_lines = index + 1, [line, lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            block = _base_block("table", "\n".join(table_lines), "python",
                                start_line=table_start, end_line=index)
            block["section_path"] = list(headings)
            blocks.append(block)
            continue
        elif not line.strip():
            flush()
        else:
            if not buffer:
                start = index + 1
            buffer.append(line)
        index += 1
    flush("code" if fence else "paragraph")
    return blocks, {"encoding": encoding}


import os
from agent_modeling.errors import ConfigurationError, DocumentParsingError


def normalize_mineru(items, *, paginated=True):
    """Content List V1: zero-based page_idx, bbox normalized to 0..1000."""
    if not isinstance(items, list):
        raise DocumentParsingError("MinerU content list must be a list")
    blocks, headings = [], []
    for index, item in enumerate(items):
        kind = item.get("type", "text")
        if kind in {"page_header", "page_footer", "page_number"}:
            continue
        source = {"parser_item": index}
        if paginated:
            source["page_index"] = item.get("page_idx")
            if item.get("bbox"):
                source.update(bbox=[v / 1000 for v in item["bbox"]],
                              coordinate_system="normalized_top_left")
        content = item.get("text", "")
        if kind == "table":
            content = "\n".join([*item.get("table_caption", []), item.get("table_body", ""),
                                  *item.get("table_footnote", [])])
        elif kind in {"image", "chart"}:
            content = "\n".join([*item.get(f"{kind}_caption", []),
                                  str(item.get("content", "")), *item.get(f"{kind}_footnote", [])])
        elif kind in {"list", "index"}:
            content = "\n".join(item.get("list_items", []))
        elif kind == "code":
            content = item.get("code_body", "")
        block = _base_block(kind, content, "mineru", **source)
        if kind == "table":
            block["footnotes"] = "\n".join(item.get("table_footnote", []))
        if "text_level" in item:
            level = max(1, int(item["text_level"]))
            headings = headings[:level - 1] + [content]
            block.update(type="heading", level=level)
        block["section_path"] = list(headings)
        if item.get("img_path"):
            block["asset_ref"] = item["img_path"]
        if content.strip() or kind in {"table", "image", "chart"}:
            blocks.append(block)
    return blocks


def run_mineru(path, output_dir, options):
    try:
        from mineru.parser import parse, ParseResult
        from mineru.render import RenderFormat, render
        from mineru.parser.writer import FileBasedDataWriter
    except ImportError as exc:
        raise ConfigurationError("MinerU 4.x SDK is required; install requirements/documents.txt") from exc
    if options.mineru_api_url:
        from mineru.parser import MinerUApiParser
        parser = MinerUApiParser(api_url=options.mineru_api_url,
                                 api_key=os.getenv("MINERU_API_KEY", ""),
                                 tier=options.mineru_tier, include_images=True)
        result = parser.parse(str(path), **({"page_range": "all"} if path.suffix.lower() == ".pdf" else {}))
    else:
        result = parse(str(path), tier=options.mineru_tier,
                       **({"page_range": "all"} if path.suffix.lower() == ".pdf" else {}))
    if result.to_dict().get("is_full_document") is False:
        raise DocumentParsingError("MinerU returned an incomplete document")
    result.save(FileBasedDataWriter(str(output_dir)))
    # save() materializes images on a copy. Render that copy, avoiding base64
    # assets being injected into canonical JSON or the LLM-readable view.
    saved = ParseResult.from_json((output_dir / "middle_json.json").read_text(encoding="utf-8"))
    items = render(saved.middle_json, RenderFormat.CONTENT_LIST)
    return normalize_mineru(items, paginated=path.suffix.lower() == ".pdf")



def render_document(document):
    """LLM-readable view; canonical JSON retains table markup and source geometry."""
    parts = []
    for block in document["blocks"]:
        source = block["source"]
        label = f"[block_id={block['block_id']}"
        if source.get("page_index") is not None:
            label += f" page={source['page_index'] + 1}"
        label += "]"
        text = block["text"]
        if block.get("footnotes") and block["footnotes"] not in text:
            text += "\n" + block["footnotes"]
        if block["type"] == "heading":
            text = "#" * min(6, block.get("level", 1)) + " " + text
        parts.append(label + "\n" + text)
    return "\n\n".join(parts)


def build_extraction_units(documents, max_chars=6000):
    """Pack source blocks; keep tables atomic and repeat locators on long prose."""
    units, pending, size = [], [], 0
    for document in documents:
        for block in document["blocks"]:
            if not block["text"].strip():
                continue
            heading = " / ".join(block.get("section_path", []))
            prefix = f"报告：{document['name']}\n章节：{heading}\n[block_id={block['block_id']}]\n"
            body = block["text"]
            if block.get("footnotes") and block["footnotes"] not in body:
                body += "\n" + block["footnotes"]
            # Never fragment a table or sever its merged-cell HTML.
            pieces = [body] if block["type"] == "table" else [body[i:i + max_chars] for i in range(0, len(body), max_chars)]
            for piece in pieces:
                entry = prefix + piece
                if pending and size + len(entry) > max_chars:
                    units.append("\n\n".join(pending))
                    pending, size = [], 0
                pending.append(entry)
                size += len(entry) + 2
        if pending:
            units.append("\n\n".join(pending))
            pending, size = [], 0
    return units


from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import tempfile
from agent_modeling.config import RUNTIME_ROOT, SESSIONS_ROOT
from agent_modeling.errors import ConfigurationError, DocumentParsingError, FileValidationError

SCHEMA_VERSION = 1


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _default_cache_root(path, cache_key):
    """Keep session caches shallow, leaving room for parser artifacts on Windows."""
    try:
        relative = path.relative_to(SESSIONS_ROOT.resolve())
    except ValueError:
        relative = None
    if relative is not None and len(relative.parts) > 1:
        return SESSIONS_ROOT / relative.parts[0] / "documents"
    nearby = path.parent / "parsed_documents"
    if len(str(nearby / cache_key)) > 180:
        scope = hashlib.sha256(str(path.parent).encode("utf-8")).hexdigest()[:16]
        return RUNTIME_ROOT / "document_cache" / scope
    return nearby


def parse_document(path, *, cache_dir=None, options=None, mineru_runner=None):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix.lower() not in DOCUMENT_SUFFIXES:
        raise FileValidationError("Unsupported or missing report document")
    options = options or ParsingOptions.from_env()
    if options.mineru_tier not in {"flash", "basic", "standard", "advanced"}:
        raise ConfigurationError("Invalid DOCUMENT_MINERU_TIER")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint = {"schema": SCHEMA_VERSION, "options": asdict(options), "suffix": path.suffix.lower(),
                   "name": path.name, "mineru": _package_version("mineru"),
                   "cache_revision": os.getenv("DOCUMENT_PARSER_CACHE_REVISION", "1")}
    cache_key = hashlib.sha256((digest + json.dumps(fingerprint, sort_keys=True)).encode()).hexdigest()
    root = Path(cache_dir) if cache_dir is not None else _default_cache_root(path, cache_key)
    target = root / cache_key
    cache_file = target / "document.json"
    if mineru_runner is None and cache_file.exists():
        try:
            document = json.loads(cache_file.read_text(encoding="utf-8"))
            if (isinstance(document, dict) and document.get("schema_version") == SCHEMA_VERSION
                    and document.get("sha256") == digest and isinstance(document.get("blocks"), list)
                    and document["blocks"] and isinstance(document.get("llm_text"), str)
                    and isinstance(document.get("warnings"), list)):
                return document
        except (ValueError, OSError):
            pass
    document = {"schema_version": SCHEMA_VERSION, "document_id": f"doc_{cache_key[:16]}",
                "name": path.name, "sha256": digest, "blocks": [], "warnings": [],
                "parser_config": fingerprint, "artifact_dir": str(target)}
    temporary = None
    try:
        target.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in SIMPLE_SUFFIXES:
            blocks, metadata = parse_simple(path)
            document["metadata"] = metadata
        else:
            blocks = (mineru_runner or run_mineru)(path, target / "mineru", options)
        if not blocks or not any(block["text"].strip() for block in blocks):
            raise DocumentParsingError("Document contains no readable content")
        for index, block in enumerate(blocks):
            block["block_id"] = f"{document['document_id']}_b{index + 1:05d}"
        document["blocks"] = blocks
        document["llm_text"] = render_document(document)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target,
                                         prefix="doc-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, indent=2)
        temporary.replace(cache_file)
        return document
    except (ConfigurationError, FileValidationError, DocumentParsingError):
        raise
    except Exception as exc:
        raise DocumentParsingError("Document conversion failed", details=str(exc)) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # A cleanup failure must not mask the parsing/write exception.
