verify:
	bash verify.sh

demo:
	python -m emberline.demo --scenario ridgeline

test:
	python -m pytest -q

# One-command MVP: verify (lint + tests + smoke), canonical instrumented demo,
# pitch assets, and a dated summary in docs/MVP_RUN.md. Idempotent.
mvp:
	python -m emberline.mvp

.PHONY: verify demo test mvp
