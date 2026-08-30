import importlib.util
import argparse
import unittest
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("workload.py")
_SPEC = importlib.util.spec_from_file_location("hicache_l2_workload", _MODULE_PATH)
workload = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(workload)


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=False):
        return iter(self.lines)


class FakeSession:
    def __init__(self, response):
        self.response = response

    def post(self, *args, **kwargs):
        return self.response


class WorkloadTest(unittest.TestCase):
    def test_formal_run_requires_a_commit_sha(self):
        args = argparse.Namespace(
            num_trials=1,
            prefix_tokens=1,
            tail_tokens=1,
            max_new_tokens=1,
            evict_request_tokens=1,
            evict_factor=1.1,
            evict_total_tokens=None,
            backup_wait=0,
            min_host_hit_ratio=0.9,
            sglang_commit=None,
            dry_run=False,
        )
        with self.assertRaises(ValueError):
            workload.validate_args(args)

    def test_formal_run_accepts_a_valid_commit_sha(self):
        args = argparse.Namespace(
            num_trials=1,
            prefix_tokens=1,
            tail_tokens=1,
            max_new_tokens=1,
            evict_request_tokens=1,
            evict_factor=1.1,
            evict_total_tokens=None,
            backup_wait=0,
            min_host_hit_ratio=0.9,
            sglang_commit="0123456789abcdef0123456789abcdef01234567",
            dry_run=False,
        )
        workload.validate_args(args)

    def test_exact_token_ids(self):
        self.assertEqual(
            workload.build_exact_token_ids([1, 2, 3], 8),
            [1, 2, 3, 1, 2, 3, 1, 2],
        )

    def test_measurement_plan_is_reproducible_and_balanced(self):
        first = workload.build_measurement_plan(8, 31600)
        second = workload.build_measurement_plan(8, 31600)
        self.assertEqual(first, second)
        self.assertEqual(first.count(["R", "P"]), 4)
        self.assertEqual(first.count(["P", "R"]), 4)

    def test_metric_total_aggregates_labels_and_timestamp(self):
        metrics = (
            'sglang:load_back_tokens_total{pool="kv"} 12\n'
            'sglang:load_back_tokens_total{pool="draft"} 3 123456\n'
        )
        self.assertEqual(
            workload.metric_total(metrics, "sglang:load_back_tokens_total"), 15
        )

    def test_cache_validation_distinguishes_host_and_cold(self):
        reload = {
            "rid": "R",
            "input_token_count": 100,
            "cached_tokens_details": {"device": 0, "host": 95, "storage": 0},
        }
        cold = {
            "rid": "P",
            "input_token_count": 100,
            "cached_tokens_details": {"device": 0, "host": 0, "storage": 0},
        }
        workload.validate_cache_result("R", reload, 0.9)
        workload.validate_cache_result("P", cold, 0.9)
        with self.assertRaises(RuntimeError):
            workload.validate_cache_result("P", reload, 0.9)

    def test_counter_delta_treats_unmaterialized_before_as_zero(self):
        before = {name: None for name in workload._COUNTER_NAMES}
        after = {name: 5.0 for name in workload._COUNTER_NAMES}
        self.assertEqual(
            workload.counter_deltas(before, after),
            {name: 5.0 for name in workload._COUNTER_NAMES},
        )

    def test_generate_parses_token_event_and_cache_details(self):
        event = (
            b'data: {"text":"x","meta_info":{"completion_tokens":1,'
            b'"prompt_tokens":8,"cached_tokens":7,'
            b'"cached_tokens_details":{"device":0,"host":7}}}'
        )
        client = workload.SGLangClient("http://unused", 1)
        client.session = FakeSession(FakeResponse([event, b"data: [DONE]"]))
        result = client.generate([1] * 8, "salt", "rid", 1)
        self.assertEqual(result["output_token_events"], 1)
        self.assertEqual(
            result["cached_tokens_details"],
            {"device": 0, "host": 7, "storage": 0},
        )

    def test_generate_rejects_malformed_sse_json(self):
        client = workload.SGLangClient("http://unused", 1)
        client.session = FakeSession(
            FakeResponse([b"data: {bad-json}", b"data: [DONE]"])
        )
        with self.assertRaises(RuntimeError):
            client.generate([1], "salt", "rid", 1)


if __name__ == "__main__":
    unittest.main()
