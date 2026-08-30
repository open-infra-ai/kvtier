.PHONY: test audit

# 优先使用基准客户端的隔离环境；不存在时回退系统 python3
PYTHON ?= $(shell if [ -x bench/repro-31600/.venv/bin/python ]; then \
	echo bench/repro-31600/.venv/bin/python; else echo python3; fi)

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover \
		-s bench/repro-31600 -p 'test_*.py' -v

audit:
	@test -n "$(RESULT)" || (echo '用法：make audit RESULT=results/results-<UTC>.json' >&2; exit 2)
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) bench/repro-31600/validate_results.py "$(RESULT)"
