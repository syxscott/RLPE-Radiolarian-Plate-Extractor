.PHONY: eval-research test-units install-deps

install-deps:
	pip install -e .
	python -m spacy download en_core_web_sm

test-units:
	PYTHONPATH=src pytest tests/test_caption_fixer.py tests/test_post_process.py tests/test_prompts.py -v

eval-research:
	PYTHONPATH=src python scripts/run_research_eval.py \
		--split data/splits/research_v1.json \
		--bootstrap-samples 1000 \
		--folds 5 \
		--output data/snapshot/$$(date +%Y-%m-%d)/f1.json