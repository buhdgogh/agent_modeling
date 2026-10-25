"""Production nodes with real LangGraph/SQLite and scripted specialist services."""
import ast
import csv
import io
import uuid
from functools import wraps
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
from threading import Barrier
import time
import typing
import unittest
from unittest.mock import Mock
from pydantic import BaseModel
from agent_modeling import errors, fusion
from agent_modeling import execution
from agent_modeling.models import IssueRecord, EvidenceRef
from agent_modeling.documents.parser import parse_document
from agent_modeling.models import TextInfoResult, ImageAnalysisResult
from agent_modeling.storage.state import trusted_checkpoint_types, serialize_result_state

try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.types import Overwrite, interrupt, Command
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False

SOURCE = Path(__file__).resolve().parents[1] / 'agent_modeling' / 'orchestrator.py'
VALID_CSV = ','.join(fusion.CSV_COLUMNS) + '\nA,砂岩,1,,,,,30\n'


def load_master():
    # Exclude SDK imports/constructor only; use actual state, node and graph code.
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    master = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MasterAgent')
    master.body = [n for n in master.body if not isinstance(n, ast.FunctionDef) or n.name != '__init__']
    namespace = dict(globals())
    namespace.update(vars(typing))
    for module in (errors, fusion):
        namespace.update({name: getattr(module, name) for name in dir(module) if not name.startswith('_')})
    for name in ('TaskStore', 'ExecutionContext', 'ExecutionCancelled', 'current_execution', 'invoke_model',
                 'plain', 'artifact', 'summarize_events', 'cancel_job', 'snapshot_inputs', 'file_sha256'):
        namespace[name] = getattr(execution, name)
    namespace.update(add=__import__('operator').add, logger=Mock(),
                     ChatTongyi=Mock(), configure_tracing=lambda: None,
                     MAX_CHECKPOINT_LOGS=200, MAX_CHECKPOINT_MESSAGES=60)
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    return namespace['MasterAgent'], namespace


class OrchestrationSourceTests(unittest.TestCase):
    def test_specialists_are_not_registered_as_tools(self):
        source = SOURCE.read_text(encoding='utf-8')
        for removed in ('@tool', 'bind_tools', 'ToolMessage', '_tool_execution_node'):
            self.assertNotIn(removed, source)

    def test_pipeline_is_excluded_from_business_result(self):
        self.assertEqual(serialize_result_state({'pipeline': {'prompt': 'private'}, 'final_response': 'ok'}),
                         {'final_response': 'ok'})


@unittest.skipUnless(HAS_GRAPH, 'LangGraph/SQLite dependencies are not installed')
class MasterGraphTests(unittest.TestCase):
    def setUp(self):
        scratch = SOURCE.parents[1] / 'runtime'
        scratch.mkdir(exist_ok=True)
        self.temp = TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.workdir = self.temp.name
        cls, self.namespace = load_master()
        self.master = cls()
        self.master.routing_mode = 'fixed'
        self.master.enable_backtracking = True
        self.master.semantic_review_enabled = False
        self.master.semantic_review_model = 'test-review-model'
        self.master.model_name, self.master.api_key = 'test-model', 'test-key'
        for name in ('router_llm', 'text_info_bot', 'image_bot', 'kg_bot', 'borehole_bot'):
            setattr(self.master, name, Mock())
        self.master.text_info_bot.run.return_value = {'extraction': TextInfoResult(
            strata=[{'formation': '砂岩', 'formation_code': 'Q', 'confidence': 0.8}],
            profiles=[{'name': '标准剖面', 'formation': '砂岩', 'thickness': '30米'}])}
        self.master.image_bot.run.return_value = {'analysis': ImageAnalysisResult(summary='map', legends=[])}
        self.master.kg_bot.run.return_value = {'kg_data': {'nodes': [], 'edges': []}}
        self.master.borehole_bot.run.return_value = {'borehole_result': {'status': 'Success'}}
        connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.addCleanup(connection.close)
        self.master.checkpointer = SqliteSaver(connection, serde=JsonPlusSerializer(
            allowed_msgpack_modules=trusted_checkpoint_types()))
        self.master.checkpointer.setup()
        self.master.task_store = execution.TaskStore(Path(self.workdir) / 'tasks.sqlite')
        self.master.graph = self.master._build_graph()
        self.generate = Mock(return_value=VALID_CSV)
        def generate_grounded(llm, prompt, previous, issues):
            content = self.generate(llm, prompt, previous, issues)
            evidence = json.loads(prompt.split('证据库：')[-1])
            rows = list(csv.DictReader(io.StringIO(content))) if content.startswith(','.join(fusion.CSV_COLUMNS)) else []
            for row in rows:
                refs = [e['evidence_id'] for e in evidence if e['field'] == 'formation' and e['value'] == row['formation']]
                thickness_refs = [e['evidence_id'] for e in evidence if e['field'] == 'thickness']
                row.update(source_refs={'formation': refs, '厚度': thickness_refs},
                           value_kinds={'formation': 'reported', '厚度': 'reported' if thickness_refs else 'default'})
            return {'csv': content, 'rows': rows}
        self.namespace['generate_grounded_fusion'] = generate_grounded

    def run_task(self, task, **kwargs):
        options = dict(text='砂岩 Q 地层报告。标准剖面砂岩厚度30米。', instruction='执行分析', fixed_task_type=task,
                       workspace_dir=self.workdir, session_id='session')
        options.update(kwargs)
        return self.master.run(**options)

    def file(self, name, content=''):
        path = Path(self.workdir) / name
        path.write_text(content, encoding='utf-8')
        return str(path)

    def test_fixed_routes_and_authoritative_documents(self):
        documents = [{'document_id': 'report', 'blocks': []}]
        for task, field in [('text', 'final_text_info'), ('image', 'final_image_analysis'), ('kg', 'final_kg')]:
            with self.subTest(task=task):
                result = self.run_task(task, image_path=self.file('map.png'), parsed_documents=documents)
                self.assertIsNotNone(result[field])
                self.assertEqual(set(result['node_statuses']), {'decider', f'{task}_agent'})
        self.master.text_info_bot.run.assert_called_once_with(
            '砂岩 Q 地层报告。标准剖面砂岩厚度30米。', instruction='执行分析', documents=documents)
        self.master.image_bot.run.assert_called_once()
        self.master.kg_bot.run.assert_called_once()
        self.master.router_llm.invoke.assert_not_called()
        self.master.borehole_bot.run.assert_not_called()

    def test_parallel_extraction_and_join(self):
        barrier, completed = Barrier(2, timeout=5), []
        text, image = self.master.text_info_bot.run.return_value, self.master.image_bot.run.return_value
        def extract(kind, output):
            barrier.wait()  # Sequential extraction times out here.
            completed.append(kind)
            return output
        self.master.text_info_bot.run.side_effect = lambda *a, **k: extract('text', text)
        self.master.image_bot.run.side_effect = lambda *a, **k: extract('image', image)
        def generate(*args):
            self.assertCountEqual(completed, ['text', 'image'])
            return VALID_CSV
        self.generate.side_effect = generate
        result = self.run_task('fusion_borehole', image_path=self.file('map.png'))
        self.assertEqual(result['final_boreholes']['status'], 'PartialSuccess')
        self.assertEqual(result['errors'], [])
        self.assertTrue(Path(result['final_boreholes']['csv_file_path']).is_file())
        self.assertIn('borehole_text', result['node_statuses'])
        self.assertIn('borehole_image', result['node_statuses'])
        self.generate.assert_called_once()

    def test_text_evidence_warning_completes_without_human_confirmation(self):
        self.master.text_info_bot.run.return_value = {'extraction': TextInfoResult(
            strata=[{'formation': '无直接依据的地层'}],
            profiles=[{'formation': '无直接依据的地层', 'thickness': '20米'}])}
        result = self.run_task('text')
        self.assertEqual(result['task_record']['status'], 'completed')
        self.assertEqual(result['task_record']['resume_stage'], '')
        self.assertEqual(len(result['final_text_info']['strata']), 1)
        self.assertEqual(len(result['final_text_info']['profiles']), 1)
        warnings = [issue for issue in result['issues'] if issue['code'] == 'unverified_extraction']
        self.assertTrue(warnings)
        self.assertTrue(all(issue['status'] == 'open' for issue in warnings))
        self.assertEqual(result['pending_confirmations'], [])
        saved = self.master.get_task(result['task_record']['task_id'], session_id='session')
        self.assertEqual(saved['status'], 'completed')
        self.assertEqual(saved['interrupts'], [])
        self.assertNotIn('等待', result['final_response'])
        self.generate.assert_not_called()
        self.master.borehole_bot.run.assert_not_called()

    def test_reextraction_only_enters_confirmation_for_fusion_tasks(self):
        self.assertEqual(self.master._route_reextracted({'task_type': 'text'}), 'finish')
        self.assertEqual(self.master._route_reextracted({'task_type': 'image'}), 'finish')
        self.assertEqual(self.master._route_reextracted({'task_type': 'fusion_borehole'}),
                         'prepare_alignment_confirmation')

    def test_missing_optional_image_does_not_block_join(self):
        result = self.run_task('fusion_borehole')
        self.assertEqual(result['node_statuses']['borehole_image']['status'], 'skipped')
        self.assertEqual(result['final_boreholes']['status'], 'PartialSuccess')

    def test_validation_feedback_loop_then_spatial(self):
        self.generate.side_effect = [VALID_CSV.replace(',30', ',-2'), VALID_CSV]
        result = self.run_task('fusion_borehole', file_path=self.file('bounds.shp'))
        self.assertEqual(result['final_boreholes']['status'], 'Success')
        self.assertEqual(self.generate.call_count, 2)
        args = self.generate.call_args_list[1].args
        self.assertIn('-2', args[2])
        self.assertEqual(args[3][0]['code'], 'invalid_thickness')
        self.assertEqual(len(result['final_boreholes']['fusion_validation']), 2)
        self.assertEqual(result['errors'], [])
        self.master.borehole_bot.run.assert_called_once()

    def test_repair_exhaustion_and_disabled_backtracking(self):
        for enabled, attempts in [(True, 3), (False, 1)]:
            with self.subTest(backtracking=enabled):
                self.master.enable_backtracking = enabled
                self.generate.reset_mock()
                self.generate.return_value = 'invalid'
                result = self.run_task('fusion_borehole', file_path=self.file('bounds.shp'))
                self.assertEqual(self.generate.call_count, attempts)
                self.assertEqual(result['final_boreholes']['status'], 'NeedsReview')
                self.assertEqual(result['errors'][0]['node'], 'validation')
                self.master.borehole_bot.run.assert_not_called()

    def test_invalid_serialization_is_repaired(self):
        self.generate.side_effect = [fusion.FusionOutputError('not JSON', 'broken'), VALID_CSV]
        result = self.run_task('fusion_borehole')
        self.assertEqual(result['final_boreholes']['status'], 'PartialSuccess')
        self.assertEqual(self.generate.call_args_list[1].args[3][0]['code'], 'invalid_model_output')

    def test_supplied_csv_bypasses_extraction_and_generation(self):
        csv, shp, tif = self.file('config.csv', VALID_CSV), self.file('bounds.shp'), self.file('dem.tiff')
        regions = self.file('regions_ui.json', '{}')
        result = self.run_task('fusion_borehole', file_path='|'.join([csv, shp, tif, regions]))
        self.assertEqual(result['final_boreholes']['status'], 'Success')
        self.master.text_info_bot.run.assert_not_called()
        self.master.image_bot.run.assert_not_called()
        self.generate.assert_not_called()
        self.master.borehole_bot.run.assert_called_once_with(
            text='', file_paths='|'.join([result['pipeline'][key] for key in ('shp', 'tif', 'csv', 'regions')]),
            output_dir=result['workspace_dir'])

    def test_invalid_supplied_csv_is_never_rewritten(self):
        csv = self.file('config.csv', 'bad')
        result = self.run_task('fusion_borehole', file_path=csv)
        self.assertEqual(result['final_boreholes']['status'], 'NeedsReview')
        self.assertEqual(Path(csv).read_text(), 'bad')
        self.generate.assert_not_called()
        self.master.borehole_bot.run.assert_not_called()

    def test_failed_extraction_preserves_other_branch_and_stops_fusion(self):
        self.master.text_info_bot.run.side_effect = errors.ModelInvocationError('offline')
        result = self.run_task('fusion_borehole', image_path=self.file('map.png'))
        self.assertEqual(result['final_boreholes']['status'], 'Error')
        self.assertEqual(result['node_statuses']['borehole_text']['status'], 'failed')
        self.assertEqual(result['node_statuses']['borehole_image']['status'], 'success')
        self.assertEqual(result['errors'][0]['code'], 'MODEL_INVOCATION_ERROR')
        self.generate.assert_not_called()

    def test_no_sources_and_missing_explicit_files_stop(self):
        for kwargs in [dict(text=''), dict(file_path=str(Path(self.workdir) / 'missing.csv'))]:
            with self.subTest(kwargs=kwargs):
                result = self.run_task('fusion_borehole', **kwargs)
                self.assertEqual(result['final_boreholes']['status'], 'Error')
                self.generate.assert_not_called()
                self.master.borehole_bot.run.assert_not_called()

    def test_nested_image_and_spatial_errors_are_reported(self):
        self.master.image_bot.run.return_value = {'analysis': ImageAnalysisResult(
            summary='error', error_code='MODEL_INVOCATION_ERROR')}
        result = self.run_task('image', image_path=self.file('map.png'))
        self.assertEqual(result['errors'][0]['code'], 'MODEL_INVOCATION_ERROR')
        self.master.borehole_bot.run.return_value = {'borehole_result': {
            'status': 'Error', 'error_code': 'GEOSPATIAL_ERROR'}}
        result = self.run_task('fusion_borehole', file_path=self.file('bounds.shp'))
        self.assertEqual(result['node_statuses']['spatial']['status'], 'failed')
        self.assertEqual(result['errors'][0]['code'], 'GEOSPATIAL_ERROR')

    def test_dynamic_route_retries_invalid_model_decision(self):
        self.master.routing_mode = 'dynamic'
        self.master.router_llm.invoke.side_effect = [
            AIMessage(content='{"task_type":"unknown","reason":"bad"}'),
            AIMessage(content='```json\n{"task_type":"text","reason":"抽取报告"}\n```')]
        result = self.run_task('')
        self.assertEqual(result['task_type'], 'text')
        self.assertEqual(result['node_statuses']['decider']['retry_count'], 1)
        self.master.text_info_bot.run.assert_called_once()

    def test_chat_invalid_fixed_route_and_router_failure(self):
        self.assertEqual(self.run_task('text', instruction='')['next_step'], 'general_chat')
        self.assertEqual(self.run_task('unknown')['next_step'], 'error')
        self.master.routing_mode = 'dynamic'
        self.master.router_llm.invoke.return_value = AIMessage(content=json.dumps(
            dict(task_type='chat', reason='问答', answer='你好')))
        self.assertEqual(self.run_task('')['final_response'], '你好')
        self.master.router_llm.invoke.side_effect = RuntimeError('offline')
        result = self.run_task('')
        self.assertEqual(result['errors'][0]['code'], 'MODEL_INVOCATION_ERROR')
        self.assertEqual(result['node_statuses']['decider']['retry_count'], 2)
        self.master.text_info_bot.run.assert_not_called()

    def test_checkpoint_compaction_and_next_turn_reset(self):
        self.generate.return_value = 'invalid'
        failed = self.run_task('fusion_borehole')
        self.assertTrue(failed['errors'])
        task_id = failed['task_record']['task_id']
        config = {'configurable': {'thread_id': f'session:{task_id}'}}
        checkpoint = self.master.graph.get_state(config).values
        self.assertTrue(checkpoint['pipeline'])
        self.assertEqual(checkpoint['final_boreholes']['status'], 'NeedsReview')
        self.assertEqual(checkpoint['parsed_documents'], [])
        success = self.run_task('text')
        self.assertEqual(success['errors'], [])
        self.assertEqual(set(success['node_statuses']), {'decider', 'text_agent'})
        self.assertIsNone(success['final_boreholes'])
        self.assertEqual(success['pipeline'], {})
        self.assertFalse(any(m.tool_calls for m in success['messages'] if isinstance(m, AIMessage)))
        config = {'configurable': {'thread_id': f"session:{success['task_record']['task_id']}"}}
        checkpoint = self.master.graph.get_state(config).values
        self.assertEqual(checkpoint['pipeline'], {})
        self.master.reset_thread('session')
        self.assertEqual(self.master.graph.get_state(config).values, {})


if __name__ == '__main__':
    unittest.main()
