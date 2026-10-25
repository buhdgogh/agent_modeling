"""Offline behavioral tests; scripted model, no SDK or credentials required."""
import ast
import concurrent.futures
import json
import logging
from pathlib import Path
import re
import time
from types import SimpleNamespace
from typing import Any, List, Optional
import unittest
from unittest.mock import patch

from pydantic import BaseModel, Field, ValidationError
from agent_modeling.errors import ModelInvocationError
from agent_modeling.documents.reports import ReportLibrary
from agent_modeling.documents.parser import build_extraction_units
from agent_modeling.models import StratumItem, ProfileItem, LLMExtraction, TextInfoResult
from agent_modeling.execution import invoke_model, invoke_tool, check_cancelled, worker_count, submit
from agent_modeling.fusion import ground_text_records, text_sources


class Message:
    def __init__(self, content="", **kwargs):
        self.content = content
        self.__dict__.update(kwargs)


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        return next(self.responses)


class LocalTool:
    def __init__(self, name, function):
        self.name, self.function = name, function

    def invoke(self, arguments):
        return self.function(**arguments)


def local_tools(library):
    return [LocalTool(name, getattr(library, name))
            for name in ("search_reports", "read_section", "lookup_stratum")]


def load_agent_core():
    """Compile production classes, substituting only external SDK boundaries."""
    source = Path(__file__).resolve().parents[1] / "agent_modeling" / "agents" / "text.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"TextExtractionAgent"}
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = dict(BaseModel=BaseModel, Field=Field, ValidationError=ValidationError, List=List, Optional=Optional, Any=Any,
                     TextAgentState=dict, SystemMessage=Message, HumanMessage=Message,
                     ToolMessage=Message, ReportLibrary=ReportLibrary,
                     create_text_tools=local_tools, ModelInvocationError=ModelInvocationError,
                     build_extraction_units=build_extraction_units,
                     concurrent=concurrent, json=json, re=re, time=time,
                     StratumItem=StratumItem, ProfileItem=ProfileItem,
                     LLMExtraction=LLMExtraction, TextInfoResult=TextInfoResult,
                     logger=logging.getLogger("text-agent-test"))
    namespace.update(invoke_model=invoke_model, invoke_tool=invoke_tool, check_cancelled=check_cancelled,
                     worker_count=worker_count, submit=submit, ground_text_records=ground_text_records,
                     text_sources=text_sources)
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


CORE = load_agent_core()


class ReportToolsTests(unittest.TestCase):
    def test_scope_headings_and_pagination(self):
        library = ReportLibrary("--- a.txt ---\n第一章 地层\n上覆层为甲。\n厚度20米。\n第二章 构造\n断层。")
        hit = library.search_reports("厚度")["matches"][0]
        self.assertEqual(hit["report"], "a.txt")
        first = library.read_section(hit["section_id"], max_lines=2)
        self.assertEqual(first["next_line"], 3)
        self.assertIn("厚度20米", library.read_section(hit["section_id"], first["next_line"])["text"])
        other = ReportLibrary("另一会话完全无关资料")
        self.assertEqual(other.search_reports("厚度")["matches"], [])

    def test_long_line_search_returns_matching_window(self):
        library = ReportLibrary("背景" * 5000 + "目标剖面厚度21米")
        self.assertIn("目标剖面厚度21米", library.search_reports("目标剖面厚度")["matches"][0]["text"])
        self.assertEqual(library.read_section("s1")["error"], "line_too_long")

    def test_bad_section_and_empty_query(self):
        library = ReportLibrary("正文")
        self.assertEqual(library.read_section("../../other")["error"], "section_not_found")
        self.assertEqual(library.search_reports(" ")["matches"], [])

    def test_dictionary_configuration_and_alias(self):
        self.assertEqual(ReportLibrary("正文").lookup_stratum("甲")["status"], "dictionary_not_configured")
        dictionary = json.dumps([{"name": "测试地层", "aliases": ["TEST"], "source": "测试夹具"}])
        with patch.object(Path, "read_text", return_value=dictionary):
            library = ReportLibrary("正文", dictionary_path="dictionary.json")
            self.assertEqual(library.lookup_stratum("test")["matches"][0]["name"], "测试地层")
            self.assertEqual(library.lookup_stratum("不存在")["status"], "not_found")
        with patch.object(Path, "read_text", return_value='{}'):
            with self.assertRaises(ValueError):
                ReportLibrary("正文", dictionary_path="dictionary.json")

    def test_explicit_reports_are_copied(self):
        reports = {"补充": "# 地层\n测试资料"}
        library = ReportLibrary("", reports)
        reports["补充"] = "更改后"
        self.assertIn("测试资料", library.read_section("s1")["text"])


class TextAgentLoopTests(unittest.TestCase):
    def make_agent(self, responses, rounds=3):
        agent = CORE["TextExtractionAgent"].__new__(CORE["TextExtractionAgent"])
        agent.llm = ScriptedModel(responses)
        agent.model_name = "test-model"
        agent.is_deepseek = False
        agent.max_tool_rounds = rounds
        return agent

    def call(self, name, arguments, call_id="c1"):
        return Message(tool_calls=[{"name": name, "args": arguments, "id": call_id}])

    def test_model_reads_observation_then_selects_next_tool(self):
        agent = self.make_agent([
            self.call("search_reports", {"query": "厚度"}),
            self.call("read_section", {"section_id": "s1"}, "c2"),
            Message(content="资料足够", tool_calls=[]),
        ])
        messages, trace = agent._gather_context("地层甲", "核实厚度", ReportLibrary("厚度20米"), 0)
        self.assertEqual([t["tool"] for t in trace], ["search_reports", "read_section"])
        self.assertIn("厚度20米", agent.llm.seen[1][-1].content)
        self.assertEqual(agent.llm.seen[1][-1].tool_call_id, "c1")
        self.assertEqual(messages[-1].content, "资料足够")

    def test_tools_are_optional(self):
        agent = self.make_agent([Message(content="已有证据", tool_calls=[])])
        _, trace = agent._gather_context("厚度20米", "", ReportLibrary("厚度20米"), 0)
        self.assertEqual(trace, [])

    def test_unknown_and_invalid_tool_are_reported_back(self):
        agent = self.make_agent([
            Message(tool_calls=[
                {"name": "shell", "args": {}, "id": "c1"},
                {"name": "read_section", "args": {}, "id": "c2"}]),
            Message(content="停止", tool_calls=[]),
        ])
        with self.assertLogs("text-agent-test", level="WARNING"):
            _, trace = agent._gather_context("正文", "", ReportLibrary("正文"), 0)
        self.assertEqual([t["status"] for t in trace], ["unknown_tool", "error"])
        self.assertEqual(agent.llm.seen[1][-1].tool_call_id, "c2")

    def test_call_budget_completes_all_call_pairs(self):
        calls = [{"name": "lookup_stratum", "args": {"term": "甲"}, "id": f"c{i}"} for i in range(10)]
        agent = self.make_agent([Message(tool_calls=calls)])
        messages, trace = agent._gather_context("正文", "", ReportLibrary("正文"), 0)
        self.assertEqual(len([m for m in messages if hasattr(m, "tool_call_id")]), 10)
        self.assertEqual(sum(t.get("status") == "budget_exhausted" for t in trace), 2)
        self.assertEqual(trace[-1]["status"], "limit_reached")

    def test_round_limit(self):
        agent = self.make_agent([self.call("search_reports", {"query": "甲"})], rounds=1)
        _, trace = agent._gather_context("甲", "", ReportLibrary("甲"), 0)
        self.assertEqual(trace[-1]["status"], "limit_reached")

    def test_large_dictionary_observation_is_bounded(self):
        library = ReportLibrary("正文")
        library.dictionary = [{"name": "甲", "description": "说明" * 5000}]
        agent = self.make_agent([
            self.call("lookup_stratum", {"term": "甲"}), Message(content="停止", tool_calls=[]),
        ])
        messages, trace = agent._gather_context("甲", "", library, 0)
        self.assertEqual(trace[0]["status"], "result_too_large")
        self.assertLess(len(messages[3].content), 300)

    def test_explicit_legacy_mode_does_not_call_tool_model(self):
        agent = self.make_agent([])
        agent.enable_tools, agent.dictionary_path, agent.is_deepseek = False, None, False
        agent.splitter = SimpleNamespace(split_text=lambda text: [text])
        agent.structural_llm = ScriptedModel([CORE["LLMExtraction"](strata=[{'formation': '甲组'}])])
        result = agent._extract_node({"text": "正文", "instruction": ""})["extraction"]
        self.assertEqual(result.tool_trace, [])
        self.assertEqual(agent.llm.seen, [])

    def test_observations_reach_structured_extractor(self):
        agent = self.make_agent([
            self.call("search_reports", {"query": "厚度"}),
            Message(content="已核实", tool_calls=[]),
        ])
        agent.enable_tools, agent.dictionary_path, agent.is_deepseek = True, None, False
        agent.splitter = SimpleNamespace(split_text=lambda text: [text])
        agent.structural_llm = ScriptedModel([CORE["LLMExtraction"](
            strata=[CORE["StratumItem"](formation="测试地层")])])
        result = agent._extract_node({"text": "测试地层厚度20米", "instruction": "核实厚度"})["extraction"]
        self.assertEqual(result.strata[0]["formation"], "测试地层")
        self.assertEqual(len(result.tool_trace), 1)
        sent = agent.structural_llm.seen[0]
        self.assertTrue(any(hasattr(m, "tool_call_id") and "厚度20米" in m.content for m in sent))

    def test_tool_model_failure_is_not_silent_empty_success(self):
        agent = self.make_agent([])
        agent.enable_tools, agent.dictionary_path = True, None
        agent.splitter = SimpleNamespace(split_text=lambda text: [text])
        with self.assertRaises(ModelInvocationError):
            agent._extract_node({"text": "正文", "instruction": ""})

    def test_structured_documents_use_blocks_instead_of_character_splitter(self):
        document = {"document_id": "d1", "name": "报告.pdf", "llm_text": "甲组厚度20米",
                    "blocks": [{"block_id": "b1", "type": "table", "text": "甲组厚度20米",
                                "section_path": ["地层"], "source": {"page_index": 0}}]}
        agent = self.make_agent([Message(content="证据足够", tool_calls=[])])
        agent.enable_tools, agent.dictionary_path, agent.is_deepseek = True, None, False
        agent.splitter = SimpleNamespace(split_text=lambda text: self.fail("must use blocks"))
        agent.structural_llm = ScriptedModel([CORE["LLMExtraction"](strata=[{'formation': '甲组'}])])
        result = agent._extract_node({"text": document["llm_text"], "documents": [document]})
        self.assertIn("extraction", result)
        self.assertIn("b1", agent.structural_llm.seen[0][1].content)

    def test_legacy_json_evidence_is_discarded_and_regrounded_from_original_text(self):
        payload = {'profiles': [{'name': '甲剖面', 'formation': '甲组', 'thickness': '25米',
            'entity_id': 'forged', 'evidence': {'总厚度25米': ['block_id=forged_block']}}]}
        agent = self.make_agent([Message(content=json.dumps(payload, ensure_ascii=False))])
        agent.enable_tools, agent.dictionary_path, agent.is_deepseek = False, None, True
        agent.splitter = SimpleNamespace(split_text=lambda text: [text])
        result = agent._extract_node({'text': '甲剖面甲组厚度25米。', 'instruction': ''})['extraction']
        profile = result.profiles[0]
        self.assertNotEqual(profile['entity_id'], 'forged')
        self.assertNotIn('总厚度25米', profile['evidence'])
        self.assertTrue(profile['evidence']['thickness'])
        for entries in profile['evidence'].values():
            self.assertTrue(all(isinstance(entry, dict) and entry['block_id'] != 'forged_block' for entry in entries))

    def test_structured_adapter_dictionary_cannot_supply_trusted_provenance(self):
        agent = self.make_agent([])
        agent.enable_tools, agent.dictionary_path, agent.is_deepseek = False, None, False
        agent.splitter = SimpleNamespace(split_text=lambda text: [text])
        agent.structural_llm = ScriptedModel([{'strata': [{'formation': '甲组', 'entity_id': 'forged',
            'evidence': {'formation': ['block_id=forged']}}]}])
        result = agent._extract_node({'text': '甲组。', 'instruction': ''})['extraction']
        self.assertNotEqual(result.strata[0]['entity_id'], 'forged')
        self.assertTrue(all(isinstance(entry, dict) for entry in result.strata[0]['evidence']['formation']))

    def test_business_validation_error_feedback_reaches_retry(self):
        agent = self.make_agent([])
        agent.enable_tools, agent.dictionary_path, agent.is_deepseek = False, None, False
        agent.splitter = SimpleNamespace(split_text=lambda text: [text])
        agent.structural_llm = ScriptedModel([{'strata': [{'formation': '甲组', 'confidence': 2}]},
                                            {'strata': [{'formation': '甲组', 'confidence': 0.8}]}])
        with patch.object(time, 'sleep'):
            result = agent._extract_node({'text': '甲组。', 'instruction': ''})['extraction']
        self.assertEqual(result.strata[0]['confidence'], 0.8)
        self.assertTrue(any('confidence' in message.content and '格式校验' in message.content
                            for message in agent.structural_llm.seen[1]))

    def extraction_agent(self, structured, fallback):
        agent = self.make_agent([Message(content=json.dumps(fallback, ensure_ascii=False))])
        agent.enable_tools, agent.dictionary_path, agent.is_deepseek = False, None, False
        agent.splitter = SimpleNamespace(split_text=lambda text: [text])
        agent.structural_llm = ScriptedModel([structured])
        return agent

    def test_schema_instead_of_data_retries_with_json_business_template(self):
        agent = self.extraction_agent({'properties': {'strata': {'type': 'array'}}},
                                      {'strata': [{'formation': '甲组'}]})
        with patch.object(time, 'sleep'), self.assertLogs('text-agent-test', level='WARNING'):
            result = agent._extract_node({'text': '甲组厚20米。'})['extraction']
        self.assertEqual(result.strata[0]['formation'], '甲组')
        self.assertIn('JSON', agent.llm.seen[0][0].content)

    def test_malformed_schema_then_empty_is_failure(self):
        agent = self.extraction_agent({'properties': {}}, {'strata': [], 'profiles': []})
        with patch.object(time, 'sleep'), self.assertLogs('text-agent-test', level='WARNING'):
            with self.assertRaises(ModelInvocationError):
                agent._extract_node({'text': '甲组厚20米。'})

    def test_structured_parser_failure_uses_plain_json_fallback(self):
        agent = self.extraction_agent(None, {'strata': [{'formation': '甲组'}]})
        with patch.object(agent.structural_llm, 'invoke', side_effect=ValueError('Invalid tool JSON')):
            with patch.object(time, 'sleep'), self.assertLogs('text-agent-test', level='WARNING'):
                result = agent._extract_node({'text': '甲组厚20米。'})['extraction']
        self.assertEqual(result.strata[0]['formation'], '甲组')

    def test_empty_extraction_gets_one_recheck(self):
        agent = self.extraction_agent({'strata': [], 'profiles': []},
                                      {'profiles': [{'formation': '甲组', 'thickness': '20米'}]})
        with patch.object(time, 'sleep'), self.assertLogs('text-agent-test', level='WARNING'):
            result = agent._extract_node({'text': '甲组厚20米。'})['extraction']
        self.assertEqual(result.profiles[0]['thickness'], '20米')

    def test_genuine_empty_is_allowed_after_two_valid_responses(self):
        agent = self.extraction_agent({'strata': [], 'profiles': []}, {'strata': [], 'profiles': []})
        with patch.object(time, 'sleep'), self.assertLogs('text-agent-test', level='WARNING'):
            result = agent._extract_node({'text': '会议将于明天开始。'})['extraction']
        self.assertEqual(result.strata, [])
        self.assertEqual(result.profiles, [])

    def test_unknown_entity_fields_cannot_become_all_null_record(self):
        with self.assertRaises(ValueError):
            CORE['TextExtractionAgent']._validate_extraction({'strata': [{'地层名称': '甲组'}]})


class ExtractionSchemaTests(unittest.TestCase):
    def test_model_schema_excludes_server_provenance(self):
        schema = LLMExtraction.model_json_schema()
        for definition in schema['$defs'].values():
            self.assertNotIn('evidence', definition.get('properties', {}))
            self.assertNotIn('entity_id', definition.get('properties', {}))

    def test_reported_malformed_evidence_does_not_break_business_extraction(self):
        result = LLMExtraction.model_validate({'profiles': [
            {'formation': '甲组', 'thickness': '>383.60m', 'evidence': {'总厚度 >383.60m': ['block_id=doc_b00009']}},
            {'formation': '乙组', 'thickness': '523.9m', 'evidence': {'岩性说明': ['block_id=doc_b00010']}}]})
        self.assertEqual(result.profiles[0].thickness, '>383.60m')
        self.assertEqual(result.profiles[1].thickness, '523.9m')
        self.assertNotIn('evidence', result.profiles[0].model_dump())
        with self.assertRaises(ValidationError):
            ProfileItem(evidence={'thickness': ['not a server evidence dictionary']})
        with self.assertRaises(ValidationError):
            LLMExtraction(profiles=[{'thickness': {'not': 'a string'}}])


if __name__ == "__main__":
    unittest.main()
