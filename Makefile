.PHONY: clean lint data features train predict
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = dysplasia_detection
# Custom PATH for Python Virtual Environments
PYTHON_INTERPRETER = $$HOME/pythonEnvs/.venv/bin/python3
PYTHONPATH = $(PROJECT_DIR)/src


ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

#.EXPORT_ALL_VARIABLES:
#export PYTHONPATH=$(PYTHONPATH)

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

python_path:
	@echo $$PYTHONPATH

## Make Dataset
data: python_path
	$ export PYTHONPATH=$$PYTHONPATH:$(PYTHONPATH); echo $$PYTHONPATH; $(PYTHON_INTERPRETER) 
	# py src/data/make_dataset.py data/raw/data.zip data/interim

## Split Dataset
data: python_path
	$ export PYTHONPATH=$$PYTHONPATH:$(PYTHONPATH); echo $$PYTHONPATH; $(PYTHON_INTERPRETER) 
	# py src/data/split_dataset.py data/interim data/processed --random_state 49 --train_size 0.66

## Train models
train: data
	$ export PYTHONPATH=$$PYTHONPATH:$(PYTHONPATH); echo $$PYTHONPATH; $(PYTHON_INTERPRETER) 
	# py src/models/train_model.py data/interim/ models/ src/models/logs/ 224 ResNet18 50 --random_state 321432 --learning_rate 0.00025 --n_unfreeze 2 --batch_size 512 --kf_splits 4
	# py src/models/train_model.py data/processed/ models/ src/models/logs/ 224 SqueezeNet 800 --random_state 321432 --learning_rate 0.001 --n_unfreeze 4 --batch_size 512
	# py src/models/train_model.py data/processed/ models/ src/models/logs/ 224 ConvModel 100 --random_state 321432 --learning_rate 0.0001 --batch_size 512 --kf_splits 2


## Predict
predict: train
	$ export PYTHONPATH=$$PYTHONPATH:$(PYTHONPATH); echo $$PYTHONPATH; $(PYTHON_INTERPRETER) 
	# py src/models/predict_model.py data/processed/224/train/DDH/2.jpg models/ResNet18_model.mdl ResNet18

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
