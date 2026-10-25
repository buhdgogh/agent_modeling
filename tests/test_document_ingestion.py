"""Filesystem and adapter contract tests. Heavy parsing models are mocked."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_modeling.errors import ConfigurationError, DocumentParsingError, FileValidationError
from agent_modeling.documents.parser import ParsingOptions
from agent_modeling.documents.parser import build_extraction_units
from agent_modeling.documents.parser import decode_text
from agent_modeling.documents.parser import normalize_mineru, run_mineru
from agent_modeling.documents.parser import parse_document
from agent_modeling.documents.parser import render_document
from agent_modeling.documents.reports import ReportLibrary
from agent_modeling.storage.state import serialize_result_state


class IngestionTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "runtime" / "document-tests"
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_native_markdown_headings_table_and_lines(self):
        path = self.write("report.md", "# 地层\n\n甲组。\n\n| 地层 | 厚度 |\n| --- | --- |\n| 甲组 | 20 |\n")
        result = parse_document(path)
        self.assertEqual([b["type"] for b in result["blocks"]], ["heading", "paragraph", "table"])
        table = result["blocks"][-1]
        self.assertEqual(table["source"]["start_line"], 5)
        self.assertEqual(table["section_path"], ["地层"])
        self.assertIn(table["block_id"], result["llm_text"])

    def test_native_html_retains_table_spans_and_reading_order(self):
        path = self.write("report.html", '<h1>地层</h1><p>说明。</p>\n<table><tr><td rowspan="2">甲组</td><td>20</td></tr></table><p>结尾。</p><script>忽略</script>')
        result = parse_document(path)
        self.assertEqual([b["type"] for b in result["blocks"]], ["heading", "paragraph", "table", "paragraph"])
        self.assertIn('rowspan="2"', result["blocks"][2]["text"])
        self.assertNotIn("忽略", result["llm_text"])

    def test_markdown_fenced_code_does_not_create_headings_or_tables(self):
        result = parse_document(self.write("report.md", "```python\n# 注释\n| a |\n| --- |\n```\n"))
        self.assertEqual(len(result["blocks"]), 1)
        self.assertEqual(result["blocks"][0]["type"], "code")

    def test_chinese_section_hierarchy(self):
        result = parse_document(self.write("report.txt", "第一章 地层\n第一节 甲组\n正文。"))
        self.assertEqual(result["blocks"][-1]["section_path"], ["第一章 地层", "第一节 甲组"])

    def test_encodings_no_lossy_replacement(self):
        for encoding in ("utf-8-sig", "utf-16", "utf-32", "gb18030"):
            self.assertEqual(decode_text("地层".encode(encoding))[0], "地层")
        with self.assertRaises(FileValidationError):
            decode_text(b"text\x00binary")

    def test_invalid_json_empty_and_unsupported_documents(self):
        for name, text, error in [("bad.json", "{", DocumentParsingError),
                                   ("empty.txt", " ", DocumentParsingError),
                                   ("data.csv", "a,b", FileValidationError)]:
            with self.subTest(name=name), self.assertRaises(error):
                parse_document(self.write(name, text))

    def test_cache_hit_and_content_configuration_invalidation(self):
        path = self.write("report.txt", "甲地层")
        first = parse_document(path)
        with patch("agent_modeling.documents.parser.parse_simple", side_effect=AssertionError("cache missed")):
            self.assertEqual(parse_document(path)["document_id"], first["document_id"])
        path.write_text("乙地层", encoding="utf-8")
        self.assertNotEqual(parse_document(path)["document_id"], first["document_id"])
        with patch.dict(os.environ, {"DOCUMENT_PARSER_CACHE_REVISION": "new"}):
            self.assertNotEqual(parse_document(path)["document_id"], parse_document(path, options=ParsingOptions(mineru_tier="flash"))["document_id"])

    def test_deep_session_source_uses_short_session_cache_and_reuses_it(self):
        sessions = self.root / "sessions"
        path = sessions / "10" / "runs" / "r1" / "t1" / "v1" / "sources" / "group" / "report.txt"
        path.parent.mkdir(parents=True)
        path.write_text("甲组厚度25米。", encoding="utf-8")
        with patch("agent_modeling.documents.parser.SESSIONS_ROOT", sessions):
            document = parse_document(path)
            target = Path(document["artifact_dir"])
            self.assertEqual(target.parent, sessions / "10" / "documents")
            self.assertLess(len(str(target / "doc-abcdefgh.tmp")), 260)
            self.assertTrue((target / "document.json").is_file())
            self.assertEqual(list(target.glob("*.tmp")), [])
            replacement = path.parent.parent / "other" / "report.txt"
            replacement.parent.mkdir()
            replacement.write_text("甲组厚度25米。", encoding="utf-8")
            with patch("agent_modeling.documents.parser.parse_simple", side_effect=AssertionError("cache missed")):
                self.assertEqual(parse_document(replacement)["document_id"], document["document_id"])

    def test_failed_atomic_cache_replace_cleans_temporary_file(self):
        path = self.write("report.txt", "甲组厚度25米。")
        cache = self.root / "cache"
        with patch.object(Path, "replace", side_effect=OSError("replace failed")):
            with self.assertRaises(DocumentParsingError):
                parse_document(path, cache_dir=cache)
        self.assertEqual(list(cache.glob("*/*.tmp")), [])
        self.assertEqual(list(cache.glob("*/document.json")), [])

    def test_corrupt_cache_is_rebuilt(self):
        path = self.write("report.txt", "甲地层")
        result = parse_document(path)
        (Path(result["artifact_dir"]) / "document.json").write_text("{", encoding="utf-8")
        self.assertEqual(parse_document(path)["blocks"][0]["text"], "甲地层")
        (Path(result["artifact_dir"]) / "document.json").write_text('[]', encoding="utf-8")
        self.assertEqual(parse_document(path)["blocks"][0]["text"], "甲地层")

    def test_pdf_routes_to_mineru_and_preserves_table_and_footnotes(self):
        path = self.write("report.pdf", "stand-in PDF; mocked parser")
        blocks = normalize_mineru([
            {"type": "text", "text": "地层", "text_level": 1, "page_idx": 0},
            {"type": "table", "table_body": '<table><tr><td rowspan="2">甲组20</td></tr></table>',
             "table_footnote": ["厚度单位米"], "page_idx": 0, "bbox": [100, 200, 900, 500]},
        ])
        miner = Mock(return_value=blocks)
        result = parse_document(path, mineru_runner=miner)
        miner.assert_called_once()
        self.assertTrue(all(block["backend"] == "mineru" for block in result["blocks"]))
        table = result["blocks"][1]
        self.assertEqual(table["section_path"], ["地层"])
        self.assertEqual(table["source"]["bbox"], [.1, .2, .9, .5])
        self.assertIn('rowspan="2"', table["text"])
        self.assertEqual(result["warnings"], [])
        self.assertIn("厚度单位米", result["llm_text"])
        self.assertTrue((Path(result["artifact_dir"]) / "document.json").is_file())

    def test_word_formats_route_directly_to_mineru(self):
        for suffix in (".doc", ".docx"):
            with self.subTest(suffix=suffix):
                path = self.write("report" + suffix, "stand-in")
                miner = Mock(return_value=normalize_mineru(
                    [{"type": "text", "text": "正文"}], paginated=False))
                result = parse_document(path, mineru_runner=miner)
                self.assertEqual(miner.call_args.args[0], path.resolve())
                self.assertEqual(result["blocks"][0]["backend"], "mineru")
                self.assertEqual(result["warnings"], [])

    def test_word_does_not_claim_physical_page_numbers(self):
        blocks = normalize_mineru([{"type": "text", "text": "正文", "page_idx": 0, "bbox": [1, 2, 3, 4]}], paginated=False)
        self.assertNotIn("page_index", blocks[0]["source"])
        self.assertNotIn("bbox", blocks[0]["source"])

    def test_missing_dependency_does_not_fall_back_to_empty_success(self):
        path = self.write("report.pdf", "stand-in")
        with patch("agent_modeling.documents.parser.run_mineru", side_effect=ConfigurationError("missing")):
            with self.assertRaises(ConfigurationError):
                parse_document(path)

    def test_mineru_failure_is_not_cached_as_success(self):
        path = self.write("report.docx", "stand-in")
        with self.assertRaises(DocumentParsingError):
            parse_document(path, mineru_runner=Mock(side_effect=RuntimeError("backend failure")))
        self.assertEqual(list((path.parent / "parsed_documents").glob("*/document.json")), [])

    def test_empty_mineru_output_is_not_cached_as_success(self):
        path = self.write("report.pdf", "stand-in")
        with self.assertRaises(DocumentParsingError):
            parse_document(path, mineru_runner=Mock(return_value=[]))
        self.assertEqual(list((path.parent / "parsed_documents").glob("*/document.json")), [])

    def test_successful_mineru_result_is_cached(self):
        path = self.write("report.pdf", "stand-in")
        miner = Mock(return_value=normalize_mineru([{"type": "text", "text": "正文"}]))
        with patch("agent_modeling.documents.parser.run_mineru", miner):
            first = parse_document(path)
            second = parse_document(path)
        miner.assert_called_once()
        self.assertEqual(first, second)

    def test_provenance_reaches_report_tools_and_extraction_units(self):
        document = parse_document(self.write("report.md", "# 地层\n甲组厚度20米。"))
        library = ReportLibrary(document["llm_text"], documents=[document])
        matches = library.search_reports("厚度")["matches"]
        self.assertIn("block_id", matches[0])
        self.assertEqual(matches[0]["source"]["start_line"], 2)
        read = library.read_section(matches[0]["section_id"])
        self.assertEqual(read["block_id"], matches[0]["block_id"])
        self.assertEqual(len(library.sections), 2)  # No duplicate plain-text copy.
        units = build_extraction_units([document])
        self.assertIn(matches[0]["block_id"], units[0])

    def test_table_is_not_split_by_character_budget(self):
        table = {"type": "table", "text": "<table>" + "cell" * 2000 + "</table>",
                 "source": {}, "block_id": "table_1", "section_path": ["地层"]}
        units = build_extraction_units([{"name": "报告", "blocks": [table]}], max_chars=100)
        self.assertEqual(len(units), 1)
        self.assertIn(table["text"], units[0])

    def test_canonical_document_is_not_copied_to_chat_json(self):
        self.assertEqual(serialize_result_state({"parsed_documents": [{"large": "data"}], "final_response": "ok"}),
                         {"final_response": "ok"})

    def mineru_modules(self, result, items):
        parse, render = Mock(return_value=result), Mock(return_value=items)
        api = Mock(return_value=SimpleNamespace(parse=Mock(return_value=result)))
        reader = SimpleNamespace(from_json=Mock(return_value=SimpleNamespace(middle_json="materialized")))
        modules = {
            "mineru": SimpleNamespace(),
            "mineru.parser": SimpleNamespace(parse=parse, ParseResult=reader, MinerUApiParser=api),
            "mineru.render": SimpleNamespace(RenderFormat=SimpleNamespace(CONTENT_LIST="content-list"), render=render),
            "mineru.parser.writer": SimpleNamespace(FileBasedDataWriter=lambda path: Path(path)),
        }
        return modules, parse, render, api

    def test_mineru_sdk_all_pages_and_materialized_output_contract(self):
        path = self.write("report.pdf", "stand-in")

        def save(output):
            output.mkdir(parents=True, exist_ok=True)
            (output / "middle_json.json").write_text('{}', encoding="utf-8")

        result = SimpleNamespace(to_dict=lambda: {"is_full_document": True}, save=save)
        modules, parse, render, api = self.mineru_modules(result, [{"type": "text", "text": "正文", "page_idx": 0}])
        with patch.dict(sys.modules, modules):
            blocks = run_mineru(path, self.root / "miner", ParsingOptions())
        parse.assert_called_once_with(str(path), tier="standard", page_range="all")
        render.assert_called_once_with("materialized", "content-list")
        api.assert_not_called()
        self.assertEqual(blocks[0]["source"]["page_index"], 0)

    def test_mineru_sdk_rejects_partial_output(self):
        result = SimpleNamespace(to_dict=lambda: {"is_full_document": False})
        modules, _, _, _ = self.mineru_modules(result, [])
        with patch.dict(sys.modules, modules), self.assertRaises(DocumentParsingError):
            run_mineru(self.root / "report.pdf", self.root / "miner", ParsingOptions())

    def test_mineru_api_word_has_no_invented_pdf_range(self):
        def save(output):
            output.mkdir(parents=True, exist_ok=True)
            (output / "middle_json.json").write_text('{}', encoding="utf-8")
        result = SimpleNamespace(to_dict=lambda: {}, save=save)
        modules, parse, _, api = self.mineru_modules(result, [{"type": "text", "text": "正文", "page_idx": 0}])
        with patch.dict(sys.modules, modules):
            blocks = run_mineru(self.root / "report.docx", self.root / "miner", ParsingOptions(mineru_api_url="http://localhost:8000"))
        parse.assert_not_called()
        api.return_value.parse.assert_called_once_with(str(self.root / "report.docx"))
        self.assertNotIn("page_index", blocks[0]["source"])



if __name__ == "__main__":
    unittest.main()
