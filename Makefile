verify:
	bash verify.sh

demo:
	python -m emberline.demo --scenario ridgeline

test:
	python -m pytest -q

.PHONY: verify demo test
