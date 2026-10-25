"""Report library and model-callable reading tools."""


import json
import re
from pathlib import Path


def _terms(query):
    words = re.findall(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]+", query.casefold())
    return set(words + [word[i:i + 2] for word in words if len(word) > 2
                        for i in range(len(word) - 1)])


class ReportLibrary:
    """No filesystem discovery: only documents explicitly supplied for this run."""

    def __init__(self, text, reports=None, dictionary_path=None, *, documents=None):
        self.reports = dict(reports or {})
        if text.strip() and not documents:
            parts = re.split(r"(?m)^--- ([^\n]+) ---\s*\n", text)
            if len(parts) > 1:
                if parts[0].strip():
                    self.reports.setdefault("当前文本", parts[0])
                for name, content in zip(parts[1::2], parts[2::2]):
                    if name in self.reports:
                        self.reports[name] += "\n" + content
                    else:
                        self.reports[name] = content
            else:
                self.reports.setdefault("当前报告", text)
        if not all(isinstance(k, str) and isinstance(v, str)
                   for k, v in self.reports.items()):
            raise ValueError("reports must map report names to text")
        self.sections = []
        heading = re.compile(
            r"^(?:#{1,6}\s+.+|第[一二三四五六七八九十百零〇\d]+[章节篇部分].*|"
            r"\d+(?:\.\d+)*[、.\s]\s*\S.{0,78}|[一二三四五六七八九十]+[、．].+)$")
        for report, content in self.reports.items():
            title, start, body = "正文", 1, []
            lines = content.splitlines()
            for line_no, line in enumerate(lines, 1):
                if heading.match(line.strip()):
                    if body:
                        self._add(report, title, start, body)
                    title, start, body = line.strip(), line_no, [line]
                else:
                    body.append(line)
            if body:
                self._add(report, title, start, body)
        for document in documents or []:
            for block in document["blocks"]:
                content = block["text"]
                if block.get("footnotes") and block["footnotes"] not in content:
                    content += "\n" + block["footnotes"]
                self._add(document["name"], " / ".join(block.get("section_path", [])) or "正文",
                          block["source"].get("start_line", 1), content.splitlines())
                self.sections[-1].update(block_id=block["block_id"], document_id=document["document_id"],
                                         source=block["source"], block_type=block["type"])
            # Used only when callers did not supply a primary extraction text.
            self.reports[document["document_id"]] = document["llm_text"]
        self.dictionary = []
        if dictionary_path:
            data = json.loads(Path(dictionary_path).read_text(encoding="utf-8-sig"))
            if not isinstance(data, list) or not all(
                isinstance(item, dict) and isinstance(item.get("name"), str)
                and isinstance(item.get("aliases", []), list)
                and all(isinstance(alias, str) for alias in item.get("aliases", []))
                for item in data
            ):
                raise ValueError("Dictionary must be a JSON list of records with name and optional aliases")
            self.dictionary = data

    def _add(self, report, title, start, lines):
        self.sections.append({"section_id": f"s{len(self.sections) + 1}",
                              "report": report, "title": title,
                              "start_line": start, "lines": tuple(lines)})

    def catalog(self):
        return [{"section_id": s["section_id"], "report": s["report"][:200],
                 "title": s["title"][:160], "start_line": s["start_line"]}
                for s in self.sections[:100]]

    def search_reports(self, query, limit=5):
        if not query.strip():
            return {"matches": [], "message": "请输入检索关键词"}
        terms, hits = _terms(query), []
        for section in self.sections:
            lines = section["lines"]
            # Rank passages, not just the beginning of a potentially long section.
            for offset in range(0, len(lines), 8):
                passage = "\n".join(lines[offset:offset + 8])
                for char_offset in range(0, len(passage), 1200):
                    window = passage[char_offset:char_offset + 1800]
                    score = sum(term in window.casefold() for term in terms)
                    if score:
                        hits.append({"section_id": section["section_id"],
                                     "report": section["report"], "title": section["title"],
                                     **{key: section[key] for key in ("block_id", "document_id", "source", "block_type") if key in section},
                                     "start_line": section["start_line"] + offset
                                     + passage[:char_offset].count("\n"),
                                     "text": window, "score": score})
        hits.sort(key=lambda hit: hit["score"], reverse=True)
        return {"matches": hits[:max(1, min(int(limit), 8))],
                "scope": "本次传入报告的本地关键词检索"}

    def read_section(self, section_id, start_line=1, max_lines=40):
        section = next((s for s in self.sections if s["section_id"] == section_id), None)
        if section is None:
            return {"error": "section_not_found", "sections": self.catalog()}
        offset = max(0, int(start_line) - 1)
        selected = section["lines"][offset:offset + max(1, min(int(max_lines), 80))]
        # Return a resumable cursor even for a very long individual line.
        result, used = [], 0
        for line in selected:
            if used + len(line) > 6000:
                if not result:
                    return {"section_id": section_id, "error": "line_too_long",
                            "message": "该行超过读取上限，请使用search_reports检索片段"}
                break
            result.append(line)
            used += len(line) + 1
        next_line = offset + len(result) + 1
        return {"section_id": section_id, "report": section["report"],
                **{key: section[key] for key in ("block_id", "document_id", "source", "block_type") if key in section},
                "title": section["title"], "start_line": section["start_line"] + offset,
                "text": "\n".join(result), "total_lines": len(section["lines"]),
                "next_line": next_line if next_line <= len(section["lines"]) else None}

    def lookup_stratum(self, term):
        if not self.dictionary:
            return {"matches": [], "status": "dictionary_not_configured",
                    "message": "地层词典未配置或为空；不得据此补造年代、厚度或地层对应关系"}
        term = term.strip().casefold()
        matches = [item for item in self.dictionary if term and any(
            term in name.casefold() for name in [item["name"], *item.get("aliases", [])])]
        return {"matches": matches[:5], "status": "ok" if matches else "not_found",
                "scope": "本地配置词典；词典解释不等于报告实测证据"}


def create_text_tools(library: ReportLibrary):
    """Register tools with this run's library, never mutable agent-wide report data."""
    from langchain_core.tools import tool

    @tool
    def search_reports(query: str, limit: int = 5) -> dict:
        """检索本次报告中的关键词，返回证据片段、报告名、章节ID和行号。"""
        return library.search_reports(query, limit)

    @tool
    def read_section(section_id: str, start_line: int = 1, max_lines: int = 40) -> dict:
        """按章节ID读取报告。start_line是章节内1起始行号；可用next_line继续读取。"""
        return library.read_section(section_id, start_line, max_lines)

    @tool
    def lookup_stratum(term: str) -> dict:
        """按地层名称或别名查询本地词典；未配置或未命中会明确返回空结果。"""
        return library.lookup_stratum(term)

    return [search_reports, read_section, lookup_stratum]
