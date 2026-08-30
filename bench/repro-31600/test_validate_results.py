import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("validate_results.py")
_SPEC = importlib.util.spec_from_file_location("hicache_l2_validate_results", _MODULE_PATH)
validator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validator)


def complete_record():
    result = {
        "input_token_count": 12,
        "ttft_s": 2.0,
        "total_s": 3.0,
        "output_token_events": 1,
        "cached_tokens_details": {"device": 0, "host": 12, "storage": 0},
    }
    cold = {
        **result,
        "ttft_s": 1.0,
        "cached_tokens_details": {"device": 0, "host": 0, "storage": 0},
    }
    return {
        "schema_version": 1,
        "benchmark": "sglang-hicache-l2-single-gpu",
        "run_id": "20260828T000000Z",
        "status": "complete",
        "args": {
            "sglang_commit": "0123456789abcdef0123456789abcdef01234567",
            "num_trials": 1,
            "prefix_tokens": 8,
            "tail_tokens": 4,
            "max_new_tokens": 1,
            "evict_request_tokens": 6,
            "min_host_hit_ratio": 0.9,
        },
        "server": {
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "attn_cp_size": 1,
            "speculative_algorithm": None,
            "enable_hierarchical_cache": True,
            "hicache_host_memory_mode": "cache",
            "hicache_write_policy": "write_through",
            "enable_metrics": True,
            "enable_cache_report": True,
            "hicache_storage_backend": None,
            "memory_usage": {"token_capacity": 20},
        },
        "eviction": {"num_requests": 1, "request_tokens": 6},
        "phases": {
            "W": [result],
            "E": [
                {
                    **result,
                    "input_token_count": 6,
                    "cached_tokens_details": {"device": 0, "host": 6, "storage": 0},
                }
            ],
        },
        "measurement_plan": [["R", "P"]],
        "trials": [
            {
                "trial": 0,
                "order": ["R", "P"],
                "results": {"R": result, "P": cold},
                "ttft_ratio_r_over_p": 2.0,
            }
        ],
        "metrics": {
            "counter_deltas": {
                "sglang:hicache_backup_tokens_total": 12.0,
                "sglang:evicted_tokens_total": 6.0,
                "sglang:load_back_tokens_total": 12.0,
            }
        },
        "summary": {
            "reload": {"n": 1},
            "cold_prefill": {"n": 1},
            "paired_ttft_ratio_median": 2.0,
            "scope": "single-GPU HiCache L2 microbenchmark; not an issue #31600 reproduction",
        },
    }


class ValidateResultsTest(unittest.TestCase):
    def test_complete_record(self):
        result = validator.validate_record(complete_record())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["trials"], 1)

    def test_failed_record_is_structurally_valid(self):
        result = validator.validate_record(
            {
                "schema_version": 1,
                "benchmark": "sglang-hicache-l2-single-gpu",
                "run_id": "run",
                "status": "failed",
                "error": "service unavailable",
            }
        )
        self.assertEqual(result["status"], "failed")

    def test_rejects_cold_request_with_cache_hit(self):
        record = complete_record()
        record["trials"][0]["results"]["P"]["cached_tokens_details"]["host"] = 1
        with self.assertRaises(validator.ValidationError):
            validator.validate_record(record)

    def test_rejects_malformed_nested_summary(self):
        record = complete_record()
        record["summary"]["reload"] = None
        with self.assertRaises(validator.ValidationError):
            validator.validate_record(record)

    def test_cli_reads_json_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(complete_record()), encoding="utf-8")
            self.assertEqual(validator.main([str(path)]), 0)


if __name__ == "__main__":
    unittest.main()
