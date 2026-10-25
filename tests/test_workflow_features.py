"""Evidence integrity, durable review, cancellation, accounting and concurrency."""
import concurrent.futures
import ast
import hashlib
import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from agent_modeling import execution, fusion
from agent_modeling.models import TextInfoResult
import test_orchestration as graph_tests


class EvidenceTests(unittest.TestCase):
    def sources(self):
        return fusion.collect_evidence(
            {"strata": [{"formation": "甲组", "formation_code": "Q"}],
             "profiles": [{"formation": "甲组", "name": "甲剖面", "thickness": "0.03千米"}]}, {},
            [{"document_id": "doc", "blocks": [{"block_id": "b1", "text": "甲剖面甲组 Q，厚度0.03千米。", "source": {"page_index": 2}}]}], "", "")

    def row(self, evidence):
        row = dict(zip(fusion.CSV_COLUMNS, ["part_1", "甲组", 1, "", "", "", "", 30]))
        row.update(source_refs={"formation": [next(e["evidence_id"] for e in evidence if e["field"] == "formation")],
                                "厚度": [next(e["evidence_id"] for e in evidence if e["field"] == "thickness")]},
                   value_kinds={"formation": "reported", "厚度": "reported"})
        return row

    def test_verified_location_and_unit_conversion(self):
        sources, evidence = self.sources()
        self.assertEqual(fusion.validate_grounding([self.row(evidence)], evidence, sources), [])
        self.assertTrue(all(e["document_id"] == "doc" and e["block_id"] == "b1" for e in evidence))
        self.assertTrue(all(e["source"]["page_index"] == 2 for e in evidence))

    def test_forged_ids_changed_values_and_wrong_source_kind_are_rejected(self):
        sources, evidence = self.sources()
        for change in (lambda r: r["source_refs"].update(formation=["forged"]),
                       lambda r: r.update(formation="不存在组"),
                       lambda r: r.update(厚度=300),
                       lambda r: r["value_kinds"].update(formation="image")):
            with self.subTest(change=change):
                row = self.row(evidence)
                change(row)
                self.assertTrue(fusion.validate_grounding([row], evidence, sources))

    def test_unrelated_profile_thickness_cannot_support_another_formation(self):
        sources, evidence = self.sources()
        sources["profiles"][0]["formation"] = "乙组"
        self.assertTrue(fusion.validate_grounding([self.row(evidence)], evidence, sources))

    def test_defaults_do_not_override_known_thickness(self):
        sources, evidence = self.sources()
        row = self.row(evidence)
        row["value_kinds"]["厚度"] = "default"
        self.assertTrue(fusion.validate_grounding([row], evidence, sources))
        sources["profiles"] = []
        self.assertEqual(fusion.validate_grounding([row], evidence, sources), [])
        row["厚度"] = 40
        self.assertTrue(fusion.validate_grounding([row], evidence, sources))

    def test_cooccurrence_elsewhere_in_a_document_is_not_an_entity_citation(self):
        records, evidence = fusion.ground_text_records(
            [{"formation": "甲组", "thickness": "20米"}],
            [{"document_id": "d", "block_id": "b", "source": {}, "quote": "甲组。\n乙组厚度20米。"}], "profile")
        self.assertNotIn("thickness", records[0]["evidence"])
        self.assertFalse(any(e["field"] == "thickness" for e in evidence))

    def test_ranges_approximations_and_agent_written_source_text_are_not_exact_proof(self):
        for quote in ('甲组厚度约30米', '甲组厚度20至30米', '甲组厚度30米以上'):
            self.assertFalse(fusion.supports(30, quote, '厚度'))
        sources, evidence = fusion.collect_evidence(
            {'strata': [{'formation': '虚构组'}], 'source_text': '虚构组厚度30米'}, {}, [], '', '')
        self.assertEqual(evidence, [])

    def test_age_evidence_cannot_be_borrowed_from_an_unrelated_formation(self):
        sources, evidence = fusion.collect_evidence({'strata': [
            {'formation': '甲组'}, {'formation': '乙组', 'formation_age_1': '侏罗纪'}]}, {}, [],
            '甲组。乙组形成于侏罗纪。', '')
        row = dict(zip(fusion.CSV_COLUMNS, ['part_1', '甲组', 1, '侏罗纪', '', '', '', 30]))
        row.update(source_refs={
            'formation': [next(e['evidence_id'] for e in evidence if e['field'] == 'formation' and e['value'] == '甲组')],
            'formation_age_1': [next(e['evidence_id'] for e in evidence if e['field'] == 'formation_age_1')]},
            value_kinds={'formation': 'reported', 'formation_age_1': 'reported', '厚度': 'default'})
        self.assertTrue(fusion.validate_grounding([row], evidence, sources))

    def test_image_evidence_has_image_hash_legend_and_bbox(self):
        with TemporaryDirectory(dir=graph_tests.SOURCE.parents[1] / 'runtime') as folder:
            path = Path(folder) / "map.png"
            path.write_bytes(b"fixture-image")
            sources, evidence = fusion.collect_evidence({}, {"legends": [
                {"id": 1, "formation": "甲组", "legend_text": "甲组", "text_bbox": [1, 2, 3, 4]}]}, [], "", str(path))
            self.assertEqual(evidence[0]["document_id"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(evidence[0]["source"]["bbox"], [1, 2, 3, 4])
            self.assertEqual(evidence[0]["source"]["verification"], "extraction_trace")

    def test_unaccepted_cross_modal_alignment_cannot_be_claimed(self):
        sources, evidence = self.sources()
        image = dict(evidence[0], evidence_id="image-ev", source_kind="image", field="age_text", value="侏罗纪", entity_id="legend_1")
        evidence.append(image)
        sources["legends"] = [{"id": 1, "entity_id": "legend_1", "formation": "乙组", "age_text": "侏罗纪"}]
        row = self.row(evidence)
        row["formation_age_1"] = "侏罗纪"
        row["source_refs"]["formation_age_1"] = ["image-ev"]
        row["value_kinds"]["formation_age_1"] = "image"
        self.assertIn("unaccepted_alignment", {e["code"] for e in fusion.validate_grounding([row], evidence, sources)})

    def test_explicit_superposition_evidence_prevents_reversing_the_order(self):
        sources, evidence = fusion.collect_evidence(
            {'strata': [{'formation': '甲组'}, {'formation': '乙组'}],
             'profiles': [{'formation': '甲组', 'underlying_stratum': '乙组'}]}, {}, [], '甲组下伏乙组。', '')
        rows = []
        for index, name in enumerate(('乙组', '甲组')):
            row = dict(zip(fusion.CSV_COLUMNS, ['part_1', name, index + 1, '', '', '', '', 30]))
            row.update(source_refs={'formation': [next(e['evidence_id'] for e in evidence if e['field'] == 'formation' and e['value'] == name)]},
                       value_kinds={'formation': 'reported', '厚度': 'default'})
            rows.append(row)
        self.assertIn('source_order_conflict', {issue['code'] for issue in fusion.validate_grounding(rows, evidence, sources)})


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=graph_tests.SOURCE.parents[1] / 'runtime')
        self.addCleanup(self.temp.cleanup)
        self.store = execution.TaskStore(Path(self.temp.name) / 'tasks.sqlite')

    def context(self, task="task", controls=None):
        self.store.create(task, "session")
        return execution.ExecutionContext(self.store, task, 1, controls)

    @unittest.skipUnless(graph_tests.HAS_GRAPH, 'Model callback dependencies are not installed')
    def test_usage_cost_and_each_retry_are_retained(self):
        prices = json.dumps({"test": {"input_per_million": 2, "output_per_million": 4, "currency": "CNY"}})
        with patch.dict('os.environ', APP_MODEL_PRICES_JSON=prices):
            context = self.context()
        calls = []
        class Model:
            def invoke(self, messages, config):
                calls.append(1)
                if len(calls) == 1:
                    raise TimeoutError("transient")
                return SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 50}, response_metadata={})
        with context.bind(), context.call("run", "workflow"):
            execution.invoke_model(Model(), [], model_name="test")
        events = self.store.events("task")
        finals = [e for e in events if e["kind"] == "model" and e["status"] != "started"]
        self.assertEqual([e["status"] for e in finals], ["failed", "success"])
        self.assertEqual([e["attempt"] for e in finals], [1, 2])
        self.assertEqual(len({e["call_id"] for e in finals}), 2)
        self.assertEqual(len({e['operation_id'] for e in finals}), 1)
        run = next(e for e in events if e['kind'] == 'run' and e['status'] == 'success')
        self.assertTrue(all(e['parent_call_id'] == run['call_id'] for e in finals))
        self.assertAlmostEqual(finals[1]["cost"], 0.0004)
        metrics = execution.summarize_events(events)
        self.assertEqual(metrics["model_calls"], 2)
        self.assertEqual(metrics["input_tokens"], 100)
        self.assertIsNone(metrics["total_cost"])
        self.assertGreater(metrics["wall_duration_ms"], 0)
        self.assertEqual(execution.TaskStore(self.store.path).events("task"), events)

    @unittest.skipUnless(graph_tests.HAS_GRAPH, 'Model callback dependencies are not installed')
    def test_structured_model_usage_is_collected_through_callback(self):
        context = self.context()
        class Model:
            def invoke(self, inputs, config):
                message = SimpleNamespace(usage_metadata={"input_tokens": 12, "output_tokens": 3}, response_metadata={})
                result = SimpleNamespace(generations=[[SimpleNamespace(message=message)]], llm_output={})
                config["callbacks"][0].on_llm_end(result)
                return {"structured": "no raw message"}
        with context.bind():
            execution.invoke_model(Model(), [], model_name="unpriced")
        final = self.store.events("task")[-1]
        self.assertEqual(final["input_tokens"], 12)
        self.assertIsNone(final["cost"])

    @unittest.skipUnless(graph_tests.HAS_GRAPH, 'Model callback dependencies are not installed')
    def test_shared_model_concurrency_limit_and_context_propagation(self):
        controls = execution.ExecutionController(max_model_calls=1, max_workers=2)
        contexts = [self.context(str(index), controls) for index in range(3)]
        lock, active, peak = threading.Lock(), [0], [0]
        class Model:
            def invoke(self, inputs, config):
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.03)
                with lock:
                    active[0] -= 1
                return SimpleNamespace(usage_metadata={}, response_metadata={})
        def call(context):
            with context.bind(), concurrent.futures.ThreadPoolExecutor(max_workers=2) as children:
                future = execution.submit(children, execution.invoke_model, Model(), [])
                future.result(timeout=5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as workers:
            list(workers.map(call, contexts))
        self.assertEqual(peak[0], 1)
        self.assertTrue(all(any(e["kind"] == "model" for e in self.store.events(str(i))) for i in range(3)))

    def test_cancel_while_waiting_for_slot_makes_no_provider_call(self):
        controls = execution.ExecutionController(max_model_calls=1)
        context = self.context(controls=controls)
        controls.model_slots.acquire()
        called = []
        def work():
            with context.bind():
                execution.invoke_model(SimpleNamespace(invoke=lambda *a, **k: called.append(1)), [])
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(work)
            self.store.cancel("task", "session")
            with self.assertRaises(execution.ExecutionCancelled):
                future.result(timeout=3)
        controls.model_slots.release()
        self.assertEqual(called, [])

    def test_stale_or_cross_session_claim_cannot_overwrite_a_task(self):
        self.context()
        record = self.store.get("task")
        record["status"] = "awaiting_review"
        self.store.save("task", record)
        with self.assertRaises(ValueError):
            self.store.claim("task", "other-session", 1)
        self.assertEqual(self.store.claim("task", "session", 1)["version"], 2)
        with self.assertRaises(ValueError):
            self.store.claim("task", "session", 1)

    def test_file_snapshots_remain_stable_when_uploads_are_replaced(self):
        from agent_modeling.storage import workspace
        root = Path(self.temp.name)
        with patch.object(workspace, 'SESSIONS_ROOT', root / 'uploads'):
            uploaded = SimpleNamespace(name='report.txt', getbuffer=lambda: memoryview(b'original'))
            original = workspace.save_uploaded_file(uploaded, 'session', 'batch')
            pinned = execution.snapshot_inputs({'workspace_dir': str(root / 'run'), 'document_paths': [original]})
            replacement = SimpleNamespace(name='report.txt', getbuffer=lambda: memoryview(b'replacement'))
            workspace.save_uploaded_file(replacement, 'session', 'batch')
            self.assertEqual(Path(original).read_bytes(), b'replacement')
            self.assertEqual(Path(pinned['document_paths'][0]).read_bytes(), b'original')
            self.assertEqual(pinned['source_files'][0]['sha256'], hashlib.sha256(b'original').hexdigest())

    def test_background_cancel_before_context_starts_is_version_scoped(self):
        entered, release = threading.Event(), threading.Event()
        def work(session_id, task_id, expected_version):
            entered.set()
            release.wait(timeout=5)
            context = execution.ExecutionContext(self.store, task_id, expected_version + 1)
            context.check()
        self.context('background')
        execution.launch_job('background', work, session_id='session', task_id='background', expected_version=1)
        self.assertTrue(entered.wait(timeout=3))
        self.assertFalse(execution.cancel_job('background', 'wrong-session'))
        self.assertTrue(execution.cancel_job('background', 'session'))
        release.set()
        with self.assertRaises(execution.ExecutionCancelled):
            execution.get_job('background').result(timeout=3)
        # A cancellation of version 2 must not affect a later version.
        execution.ExecutionContext(self.store, 'background', 3).check()
        execution.forget_job('background')


class UILifecycleTests(unittest.TestCase):
    def load_poll(self, future):
        source = graph_tests.SOURCE.with_name('interface.py')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'poll_active_task']
        tree.body[0].decorator_list = []
        class State(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
        state = State(active_task={'task_id': 'task', 'session_id': 'session'}, is_processing=True,
                      pending_prompt=None, uploader_key=1, master_agent=Mock())
        state.master_agent.task_store.events.return_value = []
        st, db = Mock(), Mock()
        st.session_state = state
        namespace = dict(st=st, db=db, logger=Mock(), get_job=Mock(return_value=future), forget_job=Mock(),
                         serialize_result_state=graph_tests.serialize_result_state)
        exec(compile(tree, str(source), 'exec'), namespace)
        return namespace, state, db

    def test_completed_background_result_is_persisted_once_and_unblocks_input(self):
        future = concurrent.futures.Future()
        future.set_result({'final_response': '完成', 'next_step': 'extract_text_info',
                           'task_record': {'task_id': 'task', 'status': 'completed'}})
        namespace, state, db = self.load_poll(future)
        namespace['poll_active_task']()
        self.assertFalse(state.is_processing)
        self.assertIsNone(state.active_task)
        self.assertEqual(state.uploader_key, 2)
        namespace['poll_active_task']()
        db.add_message.assert_called_once()
        self.assertEqual(db.add_message.call_args.args[4]['task_record']['task_id'], 'task')

    def test_waiting_job_keeps_ui_responsive_without_persisting_partial_result(self):
        namespace, state, db = self.load_poll(concurrent.futures.Future())
        namespace['poll_active_task']()
        self.assertTrue(state.is_processing)
        db.add_message.assert_not_called()

    def test_invalid_review_error_survives_the_rerun(self):
        future = concurrent.futures.Future()
        future.set_exception(ValueError('修正后的CSV仍未通过校验'))
        namespace, state, db = self.load_poll(future)
        namespace['poll_active_task']()
        self.assertIn('CSV', state.task_error)
        self.assertFalse(state.is_processing)
        db.add_message.assert_not_called()


@unittest.skipUnless(graph_tests.HAS_GRAPH, 'LangGraph/SQLite dependencies are not installed')
class DurableWorkflowTests(unittest.TestCase):
    setUp = graph_tests.MasterGraphTests.setUp
    run_task = graph_tests.MasterGraphTests.run_task
    file = graph_tests.MasterGraphTests.file

    def test_saved_report_retry_clears_old_failure_logs_and_exposes_safe_diagnostic(self):
        from agent_modeling.errors import DocumentParsingError
        report = self.file('report.txt', '砂岩 Q 地层报告。标准剖面砂岩厚度30米。')
        problem = DocumentParsingError('conversion failed')
        problem.__cause__ = FileNotFoundError(2, 'missing path', 'long-cache/document.tmp')
        self.namespace['parse_document'] = Mock(side_effect=problem)
        first = self.run_task('text', text='', document_paths=[report])
        self.assertEqual(first['errors'][0]['diagnostic']['cause_type'], 'FileNotFoundError')
        self.assertEqual(first['errors'][0]['diagnostic']['errno'], 2)
        second = self.master.resume(first['task_record']['task_id'], {'reason': '重新解析'}, session_id='session')
        task = self.master.get_task(second['task_record']['task_id'], session_id='session')
        self.assertEqual(len([i for i in task['issues'] if i['status'] == 'open']), 1)
        self.namespace['parse_document'] = graph_tests.parse_document
        final = self.master.resume(task['task_id'], {'reason': '修复路径后重试'}, session_id='session')
        self.assertEqual(final['task_record']['version'], 3)
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertFalse(any('DOCUMENT_PARSING_ERROR' in log for log in final['turn_thought_log']))
        self.assertEqual(final['errors'], [])

    def test_review_survives_graph_recreation_and_preserves_artifact_versions(self):
        self.generate.return_value = 'invalid'
        first = self.run_task('fusion_borehole')
        task = first['task_record']
        self.assertEqual(task['status'], 'awaiting_review')
        old = Path(task['artifacts'][0]['path']).read_bytes()
        self.master.task_store = execution.TaskStore(self.master.task_store.path)
        connection = sqlite3.connect(':memory:', check_same_thread=False)
        self.addCleanup(connection.close)
        self.master.checkpointer = graph_tests.SqliteSaver(connection, serde=graph_tests.JsonPlusSerializer(
            allowed_msgpack_modules=graph_tests.trusted_checkpoint_types()))
        self.master.checkpointer.setup()
        self.master.graph = self.master._build_graph()
        rows = [dict(zip(fusion.CSV_COLUMNS, ['part_1', '砂岩', 1, '', '', '', '', 25]))]
        final = self.master.resume(task['task_id'], {'rows': rows, 'reason': '人工核对原文厚度25米'},
                                   session_id='session', expected_version=1)
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertEqual(final['task_record']['version'], 2)
        self.assertEqual(final['final_boreholes']['status'], 'PartialSuccess')
        self.assertTrue(all(e['status'] == 'resolved' for e in final['issues']))
        self.master.text_info_bot.run.assert_called_once()
        self.generate.assert_called()
        self.assertEqual(Path(task['artifacts'][0]['path']).read_bytes(), old)
        self.assertTrue(any(e['source_kind'] == 'user' for e in final['evidence']))
        self.assertTrue(any(e['version'] == 1 for e in final['call_events']))
        self.assertTrue(any(e['version'] == 2 for e in final['call_events']))
        self.assertEqual(self.master.task_store.get(task['task_id'])['snapshot']['pipeline']['rows'][0]['厚度'], 25)
        manifest = next(a for a in final['artifacts'] if a['kind'] == 'task_result' and a['version'] == 2)
        self.assertEqual(json.loads(Path(manifest['path']).read_text(encoding='utf-8'))['task_record']['status'], 'completed')
        self.assertEqual(final['task_record']['revisions'][-1]['reason'], '人工核对原文厚度25米')

    def test_supplied_invalid_csv_is_repaired_without_mutating_original(self):
        original = self.file('bad.csv', 'invalid')
        first = self.run_task('fusion_borehole', file_path=original)
        final = self.master.resume(first['task_record']['task_id'],
            {'csv_content': graph_tests.VALID_CSV, 'reason': '修正输入配置'}, session_id='session')
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertEqual(Path(original).read_text(), 'invalid')
        self.assertEqual(fusion.validate_fusion_csv(Path(final['pipeline']['csv']).read_text(encoding='utf-8-sig')), [])
        self.master.text_info_bot.run.assert_not_called()
        self.generate.assert_not_called()

    def test_text_retry_only_reexecutes_text_stage(self):
        self.master.text_info_bot.run.side_effect = RuntimeError('temporary')
        first = self.run_task('fusion_borehole', image_path=self.file('map.png'))
        self.assertEqual(first['task_record']['resume_stage'], 'text')
        self.master.text_info_bot.run.side_effect = None
        final = self.master.resume(first['task_record']['task_id'], {'reason': '重试文本服务'}, session_id='session')
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertEqual(self.master.text_info_bot.run.call_count, 2)
        self.master.image_bot.run.assert_called_once()

    def test_bad_corrections_do_not_claim_a_version(self):
        self.generate.return_value = 'invalid'
        first = self.run_task('fusion_borehole')
        task_id = first['task_record']['task_id']
        for correction in (None, {'reason': 123}, {'reason': 'x', 'csv_content': 'invalid'}, {'reason': 'x', 'rows': 'bad'}):
            with self.subTest(correction=correction), self.assertRaises(ValueError):
                self.master.resume(task_id, correction, session_id='session')
        with self.assertRaises(ValueError):
            self.master.resume(task_id, {'reason': 'x'}, session_id='another-session')
        self.assertEqual(self.master.task_store.get(task_id)['version'], 1)

    def test_cancellation_during_model_response_stops_fusion(self):
        entered, release = threading.Event(), threading.Event()
        class Model:
            def invoke(self, inputs, config):
                entered.set()
                release.wait(timeout=5)
                return SimpleNamespace(usage_metadata={'input_tokens': 2, 'output_tokens': 1}, response_metadata={})
        expected = self.master.text_info_bot.run.return_value
        def extract(*args, **kwargs):
            execution.invoke_model(Model(), [], name='text.test', model_name='test')
            return expected
        self.master.text_info_bot.run.side_effect = extract
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.run_task, 'fusion_borehole')
            self.assertTrue(entered.wait(timeout=5))
            record = self.master.list_tasks('session')[0]
            self.assertTrue(self.master.cancel(record['task_id'], session_id='session'))
            release.set()
            final = future.result(timeout=5)
        self.assertEqual(final['task_record']['status'], 'cancelled')
        self.generate.assert_not_called()
        self.master.borehole_bot.run.assert_not_called()
        self.assertTrue(any(e['kind'] == 'model' and e['status'] == 'cancelled' for e in final['call_events']))

    def test_every_fusion_attempt_has_its_own_node_event(self):
        self.generate.return_value = 'invalid'
        result = self.run_task('fusion_borehole')
        events = [e for e in result['call_events'] if e['kind'] == 'node' and e['name'] == 'fusion' and e['status'] != 'started']
        self.assertEqual(len(events), 3)
        self.assertEqual(len({e['call_id'] for e in events}), 3)
        self.assertGreater(result['execution_metrics']['wall_duration_ms'], 0)

    def test_manual_fusion_retry_uses_prior_extraction_and_a_new_version(self):
        self.generate.return_value = 'invalid'
        first = self.run_task('fusion_borehole')
        self.generate.return_value = graph_tests.VALID_CSV
        final = self.master.resume(first['task_record']['task_id'], {'stage': 'fusion', 'reason': '重新融合'}, session_id='session')
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertEqual(self.generate.call_count, 4)
        self.master.text_info_bot.run.assert_called_once()

    def test_resume_stage_cannot_bypass_source_validation(self):
        self.generate.return_value = 'invalid'
        first = self.run_task('fusion_borehole', file_path=self.file('bounds.shp'))
        task_id = first['task_record']['task_id']
        with self.assertRaises(ValueError):
            self.master.resume(task_id, {'stage': 'finish', 'reason': '跳过'}, session_id='session')
        result = self.master.resume(task_id, {'stage': 'spatial', 'reason': '测试校验边界'}, session_id='session')
        self.assertEqual(result['task_record']['status'], 'awaiting_review')
        self.master.borehole_bot.run.assert_not_called()

    def test_supplied_configuration_is_not_sent_to_fusion_on_review(self):
        first = self.run_task('fusion_borehole', file_path=self.file('bad.csv', 'invalid'))
        with self.assertRaises(ValueError):
            self.master.resume(first['task_record']['task_id'], {'stage': 'fusion', 'reason': '重新生成'}, session_id='session')
        self.generate.assert_not_called()

    def test_later_fusion_cannot_silently_overwrite_a_manual_correction(self):
        self.master.enable_backtracking = False
        self.generate.return_value = graph_tests.VALID_CSV.replace('1,,,,,30', '1,侏罗纪,,,,30')
        first = self.run_task('fusion_borehole')
        self.assertEqual(first['task_record']['status'], 'awaiting_review')
        row = {key: first['pipeline']['rows'][0][key] for key in fusion.CSV_COLUMNS}
        row['厚度'] = 25
        self.master.enable_backtracking = True
        second = self.master.resume(first['task_record']['task_id'], {'rows': [row], 'reason': '人工确认厚度25米'}, session_id='session')
        self.assertEqual(second['task_record']['status'], 'awaiting_review')
        self.assertEqual(self.generate.call_count, 1)  # No implicit LLM rewrite after human edits.
        self.generate.return_value = graph_tests.VALID_CSV  # Deliberately ignores the manual thickness.
        third = self.master.resume(first['task_record']['task_id'], {'stage': 'fusion', 'reason': '补齐其他字段'}, session_id='session')
        self.assertEqual(third['task_record']['status'], 'awaiting_review')
        self.assertTrue(any(issue['code'] == 'manual_override_changed' and issue['status'] == 'open' for issue in third['issues']))
        self.assertEqual(third['pipeline']['manual_constraints'][0]['value'], 25)
        self.master.borehole_bot.run.assert_not_called()

    def test_document_parsing_is_a_traced_background_graph_stage(self):
        report = self.file('report.txt', '砂岩 Q 地层报告。标准剖面砂岩厚度30米。')
        result = self.run_task('text', text='', document_paths=[report])
        self.assertEqual(result['task_record']['status'], 'completed')
        self.assertTrue(any(e['name'] == 'parse_documents' and e['status'] == 'success' for e in result['call_events']))
        documents = self.master.text_info_bot.run.call_args.kwargs['documents']
        self.assertEqual(documents[0]['name'], 'report.txt')
        self.assertTrue(result['evidence'])
        self.assertTrue(all(e['source_kind'] == 'text' for e in result['evidence']))

    def test_standalone_failed_agent_can_be_retried_without_stale_open_errors(self):
        self.master.text_info_bot.run.side_effect = RuntimeError('temporary')
        first = self.run_task('text')
        self.master.text_info_bot.run.side_effect = None
        final = self.master.resume(first['task_record']['task_id'], {'reason': '服务恢复后重试'}, session_id='session')
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertTrue(all(issue['status'] == 'resolved' for issue in final['issues']))


@unittest.skipUnless(graph_tests.HAS_GRAPH, 'LangGraph dependencies are not installed')
class SemanticReviewTests(unittest.TestCase):
    def review(self, payload):
        llm = Mock()
        llm.invoke.return_value = graph_tests.AIMessage(content=json.dumps(payload, ensure_ascii=False))
        rows = [{key: '' for key in fusion.CSV_COLUMNS}]
        evidence = [{'evidence_id': 'ev_1'}]
        return fusion.review_fusion_semantics(llm, rows, evidence, {}, [], model_name='reviewer')

    def test_pass_and_evidenced_repair(self):
        self.assertEqual(self.review({'status': 'passed', 'summary': '未发现矛盾', 'findings': []})['issues'], [])
        result = self.review({'status': 'repair', 'summary': '剖面关联有误', 'findings': [
            {'row_id': '0', 'field': '厚度', 'message': '厚度属于其他剖面', 'evidence_ids': ['ev_1']}]})
        self.assertEqual(result['issues'][0]['code'], 'semantic_conflict')
        self.assertFalse(result['issues'][0]['requires_human'])

    def test_uncertainty_requires_review(self):
        result = self.review({'status': 'needs_review', 'summary': '资料歧义', 'findings': [
            {'row_id': '', 'field': '', 'message': '无法确认剖面指代', 'evidence_ids': []}]})
        self.assertTrue(result['issues'][0]['requires_human'])

    def test_rejects_invalid_review_and_fabricated_references(self):
        base = {'row_id': '0', 'field': '厚度', 'message': '冲突', 'evidence_ids': ['ev_1']}
        cases = [dict(base, evidence_ids=['fake']), dict(base, row_id='9'),
                 dict(base, field='unknown'), dict(base, evidence_ids=[])]
        for finding in cases:
            with self.subTest(finding=finding), self.assertRaises(ValueError):
                self.review({'status': 'repair', 'summary': '冲突', 'findings': [finding]})
        with self.assertRaises(ValueError):
            self.review({'status': 'passed', 'summary': '通过', 'findings': [base]})
        with self.assertRaises(ValueError):
            self.review({'status': 'repair', 'summary': '失败', 'findings': []})


@unittest.skipUnless(graph_tests.HAS_GRAPH, 'LangGraph dependencies are not installed')
class SemanticWorkflowTests(unittest.TestCase):
    setUp = graph_tests.MasterGraphTests.setUp
    run_task = graph_tests.MasterGraphTests.run_task
    file = graph_tests.MasterGraphTests.file

    def enable_review(self, responses):
        self.master.semantic_review_enabled = True
        reviewer = Mock()
        reviewer.invoke.side_effect = [graph_tests.AIMessage(content=json.dumps(r, ensure_ascii=False))
                                      if isinstance(r, dict) else r for r in responses]
        self.namespace['ChatTongyi'].return_value = reviewer
        return reviewer

    def test_semantic_pass_is_traced_and_not_repeated_in_spatial(self):
        reviewer = self.enable_review([{'status': 'passed', 'summary': '与资料相符', 'findings': []}])
        result = self.run_task('fusion_borehole', file_path=self.file('boundary.shp'))
        self.assertEqual(result['task_record']['status'], 'completed')
        reviewer.invoke.assert_called_once()
        self.assertEqual(result['pipeline']['semantic_review']['status'], 'passed')
        self.assertTrue(any(e['name'] == 'validation.semantic' and e['status'] == 'success'
                            for e in result['call_events']))
        self.master.borehole_bot.run.assert_called_once()

    def test_hard_failure_never_invokes_semantic_model(self):
        reviewer = self.enable_review([])
        self.generate.return_value = 'bad csv'
        result = self.run_task('fusion_borehole')
        reviewer.invoke.assert_not_called()
        self.assertEqual(result['task_record']['status'], 'awaiting_review')
        self.assertEqual(result['pipeline']['semantic_review']['status'], 'skipped_hard_failure')

    def test_semantic_conflict_returns_feedback_to_fusion(self):
        self.master.semantic_review_enabled = True
        calls = []
        def review(llm, rows, evidence, sources, constraints, **kwargs):
            calls.append(rows)
            issue = {'code': 'semantic_conflict', 'message': '需按原文修复剖面指代',
                     'row_id': '0', 'field': 'formation', 'evidence_ids': [evidence[0]['evidence_id']],
                     'requires_human': False}
            return {'status': 'repair' if len(calls) == 1 else 'passed', 'summary': '复核',
                    'issues': [issue] if len(calls) == 1 else [], 'findings': []}
        self.namespace['review_fusion_semantics'] = review
        result = self.run_task('fusion_borehole', file_path=self.file('boundary.shp'))
        self.assertEqual(result['task_record']['status'], 'completed')
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(self.generate.call_args.args[3][0]['code'], 'semantic_conflict')
        self.master.text_info_bot.run.assert_called_once()

    def test_ambiguity_stops_for_human_without_regeneration(self):
        reviewer = self.enable_review([{'status': 'needs_review', 'summary': '资料含糊', 'findings': [
            {'row_id': '0', 'field': 'formation', 'message': '需要确认剖面指代', 'evidence_ids': []}]}])
        result = self.run_task('fusion_borehole', file_path=self.file('boundary.shp'))
        self.assertEqual(result['task_record']['status'], 'awaiting_review')
        self.assertEqual(self.generate.call_count, 1)
        self.master.borehole_bot.run.assert_not_called()
        self.assertTrue(any(i['code'] == 'semantic_needs_review' for i in result['issues']))
        reviewer.invoke.assert_called_once()

    def test_malformed_review_blocks_spatial_and_can_be_retried(self):
        reviewer = self.enable_review([graph_tests.AIMessage(content='not json'),
                                      {'status': 'passed', 'summary': '重试完成', 'findings': []}])
        first = self.run_task('fusion_borehole', file_path=self.file('boundary.shp'))
        self.assertEqual(first['task_record']['status'], 'awaiting_review')
        self.master.borehole_bot.run.assert_not_called()
        final = self.master.resume(first['task_record']['task_id'], {'reason': '重试语义复核'}, session_id='session')
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertEqual(reviewer.invoke.call_count, 2)
        self.assertEqual(self.generate.call_count, 1)

    def test_spatial_resume_cannot_reuse_stale_semantic_pass(self):
        self.master.semantic_review_enabled = True
        row = {key: '' for key in fusion.CSV_COLUMNS}
        pipeline = {'rows': [row], 'sources': {}, 'manual_constraints': []}
        state = {'evidence': []}
        self.namespace['review_fusion_semantics'] = Mock(return_value={
            'status': 'passed', 'summary': '通过', 'issues': [], 'findings': []})
        self.master._review_semantics(state, pipeline)
        self.master._review_semantics(state, pipeline, use_cached=True)
        self.assertEqual(self.namespace['review_fusion_semantics'].call_count, 1)
        pipeline['rows'][0]['formation'] = 'changed'
        self.master._review_semantics(state, pipeline, use_cached=True)
        self.assertEqual(self.namespace['review_fusion_semantics'].call_count, 2)

    def test_provider_failure_is_not_silently_accepted(self):
        self.enable_review([RuntimeError('reviewer offline')])
        result = self.run_task('fusion_borehole', file_path=self.file('boundary.shp'))
        self.assertEqual(result['task_record']['status'], 'awaiting_review')
        self.assertEqual(result['pipeline']['semantic_review']['status'], 'unavailable')
        self.master.borehole_bot.run.assert_not_called()

    def test_direct_spatial_resume_is_blocked_by_semantic_uncertainty(self):
        self.run_initial = self.run_task('fusion_borehole', file_path=self.file('boundary.shp'))
        snapshot = self.master.task_store.get(self.run_initial['task_record']['task_id'])['snapshot']
        snapshot['pipeline']['semantic_review'] = {'status': 'passed', 'input_key': 'stale'}
        self.enable_review([{'status': 'needs_review', 'summary': '仍需确认', 'findings': [
            {'row_id': '0', 'field': 'formation', 'message': '无法确定指代', 'evidence_ids': []}]}])
        self.master.borehole_bot.reset_mock()
        update = self.master._spatial_node(snapshot)
        self.assertEqual(update['final_boreholes']['status'], 'NeedsReview')
        self.assertTrue(any(i['code'] == 'semantic_needs_review' for i in update['issues']))
        self.master.borehole_bot.run.assert_not_called()

    def test_changed_evidence_and_model_invalidate_review(self):
        self.master.semantic_review_enabled = True
        state = {'evidence': []}
        pipeline = {'rows': [], 'sources': {}, 'manual_constraints': []}
        reviewer = Mock(return_value={'status': 'passed', 'summary': '通过', 'issues': [], 'findings': []})
        self.namespace['review_fusion_semantics'] = reviewer
        self.master._review_semantics(state, pipeline)
        state['evidence'] = [{'evidence_id': 'new'}]
        self.master._review_semantics(state, pipeline, use_cached=True)
        self.master.semantic_review_model = 'another-reviewer'
        self.master._review_semantics(state, pipeline, use_cached=True)
        self.assertEqual(reviewer.call_count, 3)


@unittest.skipUnless(graph_tests.HAS_GRAPH, 'LangGraph dependencies are not installed')
class NativeConfirmationTests(unittest.TestCase):
    setUp = graph_tests.MasterGraphTests.setUp
    run_task = graph_tests.MasterGraphTests.run_task
    file = graph_tests.MasterGraphTests.file

    def default_task(self):
        self.master.text_info_bot.run.return_value = {'extraction': TextInfoResult(
            strata=[{'formation': '砂岩', 'formation_code': 'Q'}])}
        return self.run_task('fusion_borehole', text='砂岩 Q 地层报告。', file_path=self.file('bounds.shp'))

    def reply(self, result, action='accept_default', **extra):
        record = self.master.get_task(result['task_record']['task_id'], session_id='session')
        question = record['interrupts'][0]['value']['questions'][0]
        return self.master.resume(record['task_id'], {'reason': '明确确认本次假设或修改',
            'confirmation_answers': [{'question_id': question['question_id'], 'action': action, **extra}]},
            session_id='session', expected_version=record['version'])

    def test_default_pauses_natively_and_acceptance_keeps_default_provenance(self):
        first = self.default_task()
        self.assertEqual(first['task_record']['status'], 'awaiting_review')
        self.assertEqual(first['task_record']['resume_stage'], 'interrupt')
        self.master.borehole_bot.run.assert_not_called()
        config = {'configurable': {'thread_id': f"session:{first['task_record']['task_id']}"}}
        self.assertTrue(any(task.interrupts for task in self.master.graph.get_state(config).tasks))
        final = self.reply(first)
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertEqual(final['task_record']['version'], 2)
        self.assertEqual(final['pipeline']['rows'][0]['value_kinds']['厚度'], 'default')
        self.assertTrue(final['pipeline']['default_acceptances'])
        self.master.text_info_bot.run.assert_called_once()
        self.generate.assert_called_once()
        self.master.borehole_bot.run.assert_called_once()
        self.assertTrue(any(e['status'] == 'interrupted' and e['name'] == 'confirm_thickness' for e in final['call_events']))

    def test_user_entered_thickness_becomes_user_evidence_even_when_30(self):
        for thickness in (25, 30):
            with self.subTest(thickness=thickness):
                first = self.default_task()
                final = self.reply(first, 'set_thickness', thickness=thickness)
                self.assertEqual(final['task_record']['status'], 'completed')
                row = final['pipeline']['rows'][0]
                self.assertEqual(row['厚度'], thickness)
                self.assertEqual(row['value_kinds']['厚度'], 'user')
                ids = row['source_refs']['厚度']
                self.assertTrue(any(e['evidence_id'] in ids and e['source_kind'] == 'user' for e in final['evidence']))

    def test_bad_duplicate_foreign_and_stale_answers_do_not_claim_version(self):
        first = self.default_task()
        task_id = first['task_record']['task_id']
        question = first['pending_confirmations'][0]
        base = {'question_id': question['question_id'], 'action': 'accept_default'}
        for answers in ([], [base, base], [dict(base, question_id='stale')],
                        [dict(base, action='set_thickness', thickness=-1)]):
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                self.master.resume(task_id, {'reason': '测试无效输入', 'confirmation_answers': answers}, session_id='session')
        self.assertEqual(self.master.get_task(task_id, session_id='session')['version'], 1)
        with self.assertRaises(ValueError):
            self.master.resume(task_id, {'reason': '其他会话', 'confirmation_answers': [base]}, session_id='other')
        self.reply(first)
        with self.assertRaises(ValueError):
            self.master.resume(task_id, {'reason': '重复提交', 'confirmation_answers': [base]},
                               session_id='session', expected_version=1)

    def test_checkpoint_recreation_resumes_without_reextracting(self):
        first = self.default_task()
        # Rebuild the graph/checkpointer, using a separate persistent database before the pause.
        # The existing in-memory connection belongs to the fixture; export it and reopen on disk.
        path = Path(self.workdir) / 'native-checkpoint.sqlite'
        connection = sqlite3.connect(path, check_same_thread=False)
        self.addCleanup(connection.close)
        self.master.checkpointer.conn.backup(connection)
        self.master.checkpointer = graph_tests.SqliteSaver(connection, serde=graph_tests.JsonPlusSerializer(
            allowed_msgpack_modules=graph_tests.trusted_checkpoint_types()))
        self.master.checkpointer.setup()
        self.master.graph = self.master._build_graph()
        self.master.task_store = execution.TaskStore(self.master.task_store.path)
        final = self.reply(first)
        self.assertEqual(final['task_record']['status'], 'completed')
        self.master.text_info_bot.run.assert_called_once()

    def test_ambiguous_mapping_pauses_before_fusion_then_reaches_spatial_mapping(self):
        self.master.image_bot.run.return_value = {'analysis': graph_tests.ImageAnalysisResult(
            summary='map', legends=[{'id': 7, 'avg_color': [0, 120, 0], 'text_bbox': [0, 0, 10, 10]}])}
        first = self.run_task('fusion_borehole', image_path=self.file('map.png'), file_path=self.file('bounds.shp'))
        self.assertEqual(first['task_record']['resume_stage'], 'interrupt')
        self.generate.assert_not_called()
        question = first['pending_confirmations'][0]
        self.assertEqual(question['kind'], 'formation_mapping')
        candidate = question['candidates'][0]
        with self.assertRaises(ValueError):
            self.reply(first, 'select_formation', entity_id='unknown')
        final = self.reply(first, 'select_formation', entity_id=candidate['entity_id'])
        self.assertEqual(final['task_record']['status'], 'completed')
        mapping = json.loads(Path(final['pipeline']['mapping_path']).read_text(encoding='utf-8'))
        self.assertEqual(mapping, {'7': {'A': 1}})
        self.assertIn(final['pipeline']['mapping_path'], self.master.borehole_bot.run.call_args.kwargs['file_paths'])
        self.assertEqual(final['pipeline']['alignments'][0]['origin'], 'user')
        self.master.image_bot.run.assert_called_once()

    def test_new_source_invalidates_default_acceptance(self):
        pipeline = {'rows': [dict(zip(fusion.CSV_COLUMNS, ['A', '砂岩', 1, '', '', '', '', 30]))]}
        pipeline['rows'][0]['value_kinds'] = {'厚度': 'default'}
        first = fusion.default_thickness_questions(pipeline, [{'sha256': 'one'}])
        pipeline['default_acceptances'] = {first[0]['question_id']: {'reason': '接受假设'}}
        self.assertEqual(fusion.default_thickness_questions(pipeline, [{'sha256': 'one'}]), [])
        self.assertTrue(fusion.default_thickness_questions(pipeline, [{'sha256': 'two'}]))

    def test_two_successive_interrupts_resume_the_right_questions(self):
        self.master.text_info_bot.run.return_value = {'extraction': TextInfoResult(
            strata=[{'formation': '砂岩', 'formation_code': 'Q'}])}
        self.master.image_bot.run.return_value = {'analysis': graph_tests.ImageAnalysisResult(
            summary='map', legends=[{'id': 7, 'avg_color': [0, 120, 0]}])}
        first = self.run_task('fusion_borehole', text='砂岩 Q 地层报告。',
                              image_path=self.file('map.png'), file_path=self.file('bounds.shp'))
        first_question = first['pending_confirmations'][0]
        second = self.reply(first, 'select_formation', entity_id=first_question['candidates'][0]['entity_id'])
        self.assertEqual(second['task_record']['version'], 2)
        self.assertEqual(second['pending_confirmations'][0]['kind'], 'default_thickness')
        with self.assertRaises(ValueError):
            self.master.resume(second['task_record']['task_id'], {'reason': '旧回复', 'confirmation_answers': [
                {'question_id': first_question['question_id'], 'action': 'select_formation',
                 'entity_id': first_question['candidates'][0]['entity_id']}]}, session_id='session')
        final = self.reply(second)
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertEqual(final['task_record']['version'], 3)
        self.assertEqual(len(final['confirmation_history']), 2)
        self.master.text_info_bot.run.assert_called_once()
        self.master.image_bot.run.assert_called_once()
        self.generate.assert_called_once()

    def test_stage_correction_can_replace_a_pending_native_confirmation(self):
        first = self.default_task()
        rows = [{k: row[k] for k in fusion.CSV_COLUMNS} for row in first['pipeline']['rows']]
        rows[0]['厚度'] = 25
        final = self.master.resume(first['task_record']['task_id'], {'reason': '改为人工厚度', 'rows': rows},
                                   session_id='session', expected_version=1)
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertEqual(final['pipeline']['rows'][0]['value_kinds']['厚度'], 'user')
        self.assertEqual(self.master.get_task(final['task_record']['task_id'], session_id='session')['interrupts'], [])

    def test_configuration_edit_cannot_skip_unresolved_alignment(self):
        self.master.image_bot.run.return_value = {'analysis': graph_tests.ImageAnalysisResult(
            summary='map', legends=[{'id': 7, 'avg_color': [0, 120, 0]}])}
        first = self.run_task('fusion_borehole', image_path=self.file('map.png'))
        rows = [dict(zip(fusion.CSV_COLUMNS, ['A', '砂岩', 1, '', '', '', '', 25]))]
        second = self.master.resume(first['task_record']['task_id'], {'reason': '先填写配置', 'rows': rows},
                                   session_id='session')
        self.assertEqual(second['task_record']['resume_stage'], 'interrupt')
        self.assertEqual(second['pending_confirmations'][0]['kind'], 'formation_mapping')
        self.master.borehole_bot.run.assert_not_called()
        question = second['pending_confirmations'][0]
        final = self.reply(second, 'select_formation', entity_id=question['candidates'][0]['entity_id'])
        self.assertEqual(final['task_record']['status'], 'completed')
        self.generate.assert_not_called()  # Explicitly edited rows are not silently rewritten.

    def test_blank_reason_is_optional_and_deferred_thickness_stays_pending(self):
        self.master.text_info_bot.run.return_value = {'extraction': TextInfoResult(strata=[
            {'formation': '甲组', 'formation_code': 'Q1'}, {'formation': '乙组', 'formation_code': 'Q2'}])}
        self.generate.return_value = ','.join(fusion.CSV_COLUMNS) + '\nA,甲组,1,,,,,30\nA,乙组,2,,,,,30\n'
        first = self.run_task('fusion_borehole', text='甲组 Q1。乙组 Q2。', file_path=self.file('bounds.shp'))
        questions = first['pending_confirmations']
        second = self.master.resume(first['task_record']['task_id'], {'reason': None, 'confirmation_answers': [
            {'question_id': questions[0]['question_id'], 'action': 'accept_default'},
            {'question_id': questions[1]['question_id'], 'action': 'defer'}]}, session_id='session')
        self.assertEqual(second['task_record']['status'], 'awaiting_review')
        self.assertEqual(len(second['pending_confirmations']), 1)
        self.assertEqual(len(second['pipeline']['default_acceptances']), 1)
        self.master.borehole_bot.run.assert_not_called()
        final = self.reply(second)
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertEqual(len(final['pipeline']['default_acceptances']), 2)
        self.assertTrue(final['confirmation_history'][0]['reason'])

    def test_deferred_mapping_remains_native_interrupt_without_manual_reason(self):
        self.master.image_bot.run.return_value = {'analysis': graph_tests.ImageAnalysisResult(
            summary='map', legends=[{'id': 7, 'avg_color': [0, 120, 0]}])}
        first = self.run_task('fusion_borehole', image_path=self.file('map.png'))
        question = first['pending_confirmations'][0]
        second = self.master.resume(first['task_record']['task_id'], {'confirmation_answers': [
            {'question_id': question['question_id'], 'action': 'defer'}]}, session_id='session')
        self.assertEqual(second['task_record']['resume_stage'], 'interrupt')
        self.assertEqual(second['pending_confirmations'], first['pending_confirmations'])
        self.generate.assert_not_called()
        self.master.borehole_bot.run.assert_not_called()

    def test_retry_without_reason_uses_audit_default_and_text_summary_has_counts(self):
        self.master.text_info_bot.run.side_effect = RuntimeError('service offline')
        first = self.run_task('text')
        self.master.text_info_bot.run.side_effect = None
        final = self.master.resume(first['task_record']['task_id'], {}, session_id='session')
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertIn('1 个地层实体', final['final_response'])
        self.assertIn('1 条剖面信息', final['final_response'])
        self.assertTrue(final['task_record']['revisions'][-1]['reason'])

    def test_editing_one_thickness_does_not_reask_another_accepted_default(self):
        self.master.text_info_bot.run.return_value = {'extraction': TextInfoResult(strata=[
            {'formation': '甲组', 'formation_code': 'Q1'}, {'formation': '乙组', 'formation_code': 'Q2'}])}
        self.generate.return_value = ','.join(fusion.CSV_COLUMNS) + '\nA,甲组,1,,,,,30\nA,乙组,2,,,,,30\n'
        first = self.run_task('fusion_borehole', text='甲组 Q1。乙组 Q2。')
        questions = first['pending_confirmations']
        final = self.master.resume(first['task_record']['task_id'], {'confirmation_answers': [
            {'question_id': questions[0]['question_id'], 'action': 'accept_default'},
            {'question_id': questions[1]['question_id'], 'action': 'set_thickness', 'thickness': 25}]}, session_id='session')
        self.assertEqual(final['task_record']['status'], 'completed')
        self.assertEqual(final['pipeline']['rows'][0]['value_kinds']['厚度'], 'default')
        self.assertEqual(final['pipeline']['rows'][1]['value_kinds']['厚度'], 'user')
        self.assertEqual(final['pending_confirmations'], [])


class ConfirmationProtocolTests(unittest.TestCase):
    def test_explicit_spatial_mapping_follows_identity_and_part_not_legend_order(self):
        rows = [{'formation': '乙组', 'part_code': 'part_1', 'formation_code': 2},
                {'formation': '甲组', 'part_code': 'part_1', 'formation_code': 1},
                {'formation': '乙组', 'part_code': 'part_2', 'formation_code': 1}]
        self.assertEqual(fusion.spatial_legend_mapping([{'legend_id': 99, 'formation': '乙组'}], rows),
                         {'99': {'part_1': 2, 'part_2': 1}})
        with self.assertRaises(ValueError):
            fusion.spatial_legend_mapping([{'legend_id': 7, 'formation': '丙组'}], rows)
        with self.assertRaises(ValueError):
            fusion.spatial_legend_mapping([{'legend_id': 7, 'formation': '乙组'}], rows + [rows[0]])

    def test_confirmation_rejects_nonfinite_thickness_and_unknown_candidate(self):
        question = {'question_id': 'q', 'kind': 'default_thickness'}
        for thickness in (float('nan'), float('inf'), 0, -1):
            with self.subTest(thickness=thickness), self.assertRaises(ValueError):
                fusion.validate_confirmation_decisions([question], [
                    {'question_id': 'q', 'action': 'set_thickness', 'thickness': thickness}])
        with self.assertRaises(ValueError):
            fusion.validate_confirmation_decisions([{'question_id': 'm', 'kind': 'formation_mapping',
                'candidates': [{'entity_id': 'known'}]}], [{'question_id': 'm', 'action': 'select_formation', 'entity_id': 'fake'}])


class ResultPresentationTests(unittest.TestCase):
    def renderer(self):
        from contextlib import nullcontext
        source = Path(__file__).resolve().parents[1] / 'agent_modeling' / 'interface.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        names = {'result_record', 'render_text_extraction_result', 'render_analysis_result', 'safe_get'}
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        st = Mock()
        st.columns.return_value = [nullcontext(), nullcontext(), nullcontext()]
        namespace = {'st': st, 'json': json, 'os': __import__('os')}
        exec(compile(tree, str(source), 'exec'), namespace)
        return namespace, st

    def test_entities_and_profiles_are_visible_without_source_text_or_route_label(self):
        namespace, st = self.renderer()
        info = {'strata': [{'formation': '甲组', 'formation_code': 'Q', 'evidence': {'formation': [{'quote': '原文'}]}}],
                'profiles': [{'name': '甲剖面', 'formation': '甲组', 'thickness': '>383.60m', 'location': None}]}
        namespace['render_analysis_result'](None, {'final_text_info': info}, msg_id='task')
        self.assertEqual(st.dataframe.call_count, 2)
        self.assertEqual(st.dataframe.call_args_list[0].args[0][0]['地层名称'], '甲组')
        self.assertEqual(st.dataframe.call_args_list[1].args[0][0]['原文厚度'], '>383.60m')
        self.assertIsNone(st.dataframe.call_args_list[1].args[0][0]['位置'])
        exported = json.loads(st.download_button.call_args.kwargs['data'])
        self.assertEqual(exported, info)

    def test_empty_extraction_explicitly_shows_no_results(self):
        namespace, st = self.renderer()
        namespace['render_analysis_result']('extract_text_info', {'final_text_info': {'strata': [], 'profiles': []}}, msg_id='empty')
        st.dataframe.assert_not_called()
        self.assertEqual(st.info.call_count, 2)
        self.assertIn('0 个地层实体', st.caption.call_args.args[0])

    def test_configuration_only_success_is_not_reported_as_failure(self):
        namespace, st = self.renderer()
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'runtime') as directory:
            path = Path(directory) / 'formation.csv'
            path.write_text(graph_tests.VALID_CSV, encoding='utf-8')
            namespace['render_analysis_result']('generate_boreholes', {'final_boreholes': {
                'status': 'PartialSuccess', 'csv_file_path': str(path)}}, msg_id='config')
        st.error.assert_not_called()
        st.success.assert_called_once()
        self.assertEqual(st.dataframe.call_args.args[0][0]['formation'], '砂岩')
        st.download_button.assert_called_once()

    def test_knowledge_graph_content_is_shown(self):
        namespace, st = self.renderer()
        namespace['render_analysis_result']('build_kg', {'final_kg': {'nodes': [{'id': 'a'}],
            'edges': [{'source': 'a', 'target': 'b', 'relation': '下伏'}]}}, msg_id='kg')
        self.assertEqual(st.dataframe.call_count, 2)
        st.download_button.assert_called_once()


class MySQLSessionDateTests(unittest.TestCase):
    def setUp(self):
        from datetime import datetime
        source = Path(__file__).resolve().parents[1] / 'agent_modeling' / 'storage' / 'mysql.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        namespace = {'datetime': datetime, 'List': list, 'Dict': dict, 'Any': object}
        exec(compile(tree, str(source), 'exec'), namespace)
        self.parse = namespace['_session_timestamp']
        store_class = namespace['MySQLSessionStore']
        self.store = store_class.__new__(store_class)
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        self.db.row_factory = sqlite3.Row
        self.db.execute('CREATE TABLE sessions(id INTEGER, title TEXT, created_at TEXT)')
        self.db.execute('CREATE TABLE chat_history(session_id INTEGER, created_at TEXT)')
        actual_cursor = self.db.cursor()
        self.cursor = Mock()
        self.cursor.execute.side_effect = actual_cursor.execute
        self.cursor.fetchall.side_effect = lambda: [dict(row) for row in actual_cursor.fetchall()]
        self.connection = Mock()
        self.connection.cursor.return_value = self.cursor
        self.store.get_connection = Mock(return_value=self.connection)

    def test_zero_dates_empty_sessions_fallback_and_ordering(self):
        from datetime import datetime
        self.db.executemany('INSERT INTO sessions VALUES(?,?,?)', [
            (1, 'empty', '2026-01-01 10:00:00'),
            (2, 'legacy', '0000-00-00 00:00:00'),
            (3, 'active', '2026-01-01 10:00:00'),
            (4, 'unknown', None),
            (5, 'zero-message', '2026-01-02 10:00:00')])
        self.db.executemany('INSERT INTO chat_history VALUES(?,?)', [
            (2, '0000-00-00 00:00:00'), (3, '2026-01-03 12:00:00'),
            (3, '2026-00-00 00:00:00'), (3, '2026-12-00 00:00:00'),
            (3, '0000-00-00 00:00:00'), (5, '0000-00-00 00:00:00')])
        result = self.store.get_all_sessions()
        self.assertEqual([r['id'] for r in result], [3, 5, 1, 4, 2])
        self.assertEqual(result[0]['last_active'], datetime(2026, 1, 3, 12))
        self.assertEqual(result[1]['last_active'], datetime(2026, 1, 2, 10))
        self.assertIsNone(result[-1]['created_at'])
        self.assertIsNone(result[-1]['last_active'])
        self.assertNotIn('last_message_at', result[0])
        self.cursor.close.assert_called_once()
        self.connection.close.assert_called_once()

    def test_invalid_dates_remain_unknown_and_valid_precision_is_preserved(self):
        from datetime import datetime
        for value in ('0000-00-00 00:00:00', '2026-00-01 00:00:00', '2026-02-31 00:00:00', None):
            self.assertIsNone(self.parse(value))
        value = datetime(2026, 10, 25, 12, 30, 5, 123456)
        self.assertEqual(self.parse(str(value)), value)
        self.assertEqual(self.parse(value), value)

    def test_query_error_still_closes_database_resources(self):
        self.cursor.fetchall.side_effect = RuntimeError('database unavailable')
        with self.assertRaises(RuntimeError):
            self.store.get_all_sessions()
        self.cursor.close.assert_called_once()
        self.connection.close.assert_called_once()


class NativeConfirmationUITests(unittest.TestCase):
    def render(self, selected, reason=''):
        from contextlib import nullcontext
        source = Path(__file__).resolve().parents[1] / 'agent_modeling' / 'interface.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'render_native_confirmation']
        state = SimpleNamespace(is_processing=False, active_task=None, pending_prompt=None)
        st = Mock(session_state=state)
        st.form.return_value = nullcontext()
        st.selectbox.return_value = selected
        st.number_input.return_value = 30.0
        st.text_area.return_value = reason
        st.form_submit_button.return_value = True
        launch = Mock()
        namespace = dict(st=st, launch_job=launch)
        exec(compile(tree, str(source), 'exec'), namespace)
        question = {'question_id': 'default_q', 'kind': 'default_thickness', 'message': '确认默认厚度',
                    'part_code': 'part_1', 'formation': '砂岩', 'thickness': 30}
        task = {'task_id': 'task', 'version': 4, 'interrupts': [{'value': {'questions': [question]}}]}
        engine = Mock()
        namespace['render_native_confirmation'](task, engine, 'session')
        return launch, st, state

    def test_ui_can_leave_items_undecided_without_error(self):
        launch, st, state = self.render(None)
        launch.assert_not_called()
        st.error.assert_not_called()
        self.assertFalse(state.is_processing)

    def test_ui_submits_explicit_acceptance_with_current_task_version(self):
        launch, st, state = self.render('接受默认建模假设')
        submitted = launch.call_args.kwargs
        self.assertEqual(submitted['expected_version'], 4)
        self.assertEqual(submitted['corrections']['confirmation_answers'], [
            {'question_id': 'default_q', 'action': 'accept_default'}])
        self.assertEqual(submitted['corrections']['reason'], '')
        self.assertEqual(st.selectbox.call_args.kwargs['index'], 0)
        self.assertTrue(state.is_processing)


if __name__ == '__main__':
    unittest.main()
