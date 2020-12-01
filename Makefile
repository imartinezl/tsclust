.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

NAME = tsclust
TESTS = tests
PY = python3
WORKDIR = .
VENV = .venv
VENV_DIR = $(WORKDIR)/$(VENV)
VENV_BIN = $(VENV_DIR)/bin
REQUIREMENTS = requirements.txt
REQUIREMENTS_DEV = requirements_dev.txt
REQUIREMENTS_DOCS = docs/requirements.txt

BROWSER := @$(PY) -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@$(PY) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: clean clean-build clean-pyc clean-test

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

.PHONY: clean-docker clean-container clean-image docker

clean-docker: clean-container clean-image ## remove Docker image and container

clean-container: ## remove Docker container
	docker rm tsclust-dev-container

clean-image: ## remove Docker image
	docker image rm tsclust-dev-image

docker: ## docker
	docker-compose up

.PHONY: format lint pytest test-pytest test-tox coverage docs servedocs



black: venv ## installs black code formatter
	test -s $(VENV_BIN)/black || $(VENV_BIN)/pip install black

format: black ## check style with black code formatter
	$(VENV_BIN)/black $(NAME) $(TESTS)


flake8: venv ## installs flake8 code linter
	test -s $(VENV_BIN)/flake8 || $(VENV_BIN)/pip install flake8

lint: flake8  ## check style with flake8
	$(VENV_BIN)/flake8 $(NAME) $(TESTS)


pytest: venv ## installs pytest
	test -s $(VENV_BIN)/pytest || $(VENV_BIN)/pip install pytest
test-pytest: pytest ## run tests quickly with the default Python
	$(VENV_BIN)/pytest .

tox: venv ## installs tox
	test -s $(VENV_BIN)/tox || $(VENV_BIN)/pip install tox
test-tox: $(VENV_BIN)/tox ## run tests on every Python version with tox
	$(VENV_BIN)/tox

test-native: ## run tests quickly with the default Python
	$(VENV_BIN)/python setup.py test

coverage: venv ## installs coverage
	test -s $(VENV_BIN)/coverage || $(VENV_BIN)/pip install coverage
test-coverage: coverage ## check code coverage quickly with the default Python
	$(VENV_BIN)/coverage run --source tsclust setup.py test
	$(VENV_BIN)/coverage report -m
	$(VENV_BIN)/coverage html
	$(BROWSER) htmlcov/index.html

docs: install-docs-reqs ## generate Sphinx HTML documentation, including API docs
	rm -f docs/$(NAME).rst
	rm -f docs/modules.rst
	$(VENV_BIN)/sphinx-apidoc -o docs/ $(NAME)
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

.PHONY: release-test release dist install
release-test: dist ## package and upload a release
	$(VENV_BIN)/twine upload -r testpypi dist/*

release: dist ## package and upload a release
	$(VENV_BIN)/twine upload dist/*

dist: clean ## builds source and wheel package
	$(VENV_BIN)/python setup.py sdist
	$(VENV_BIN)/python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	$(VENV_BIN)/python setup.py install

.PHONY: clean-venv init-venv activate-venv show-venv debug-venv

clean-venv: ## remove virtualenv
	rm -fr $(VENV)

venv: ## create virtualenv
	test -d $(VENV) || virtualenv --clear $(VENV) --python=$(PY)

activate: venv ## activate virtualenv
	@. $(VENV_BIN)/activate

setup: clean-venv venv install-requirements freeze

show-venv: venv
	@$(VENV_BIN)/python -c "import sys; print('Python ' + sys.version.replace('\n',''))"
	@$(VENV_BIN)/pip --version
	@echo venv: $(VENVDIR)

debug-venv:
	@$(MAKE) --version
	$(info PY="$(PY)")
	$(info REQUIREMENTS="$(REQUIREMENTS)")
	$(info REQUIREMENTS_DEV="$(REQUIREMENTS_DEV)")
	$(info VENVDIR="$(VENVDIR)")
	$(info WORKDIR="$(WORKDIR)")

.PHONY: freeze install-reqs upgrade-reqs
freeze: venv ## freeze dependencies
	$(VENV_BIN)/pip freeze > requirements_freeze.txt

install-dev-reqs: venv  ## install dev-dependencies
	$(VENV_BIN)/pip install -r $(REQUIREMENTS_DEV)

upgrade-dev-reqs: venv  ## upgrade dev-dependencies
	@$(VENV_BIN)/pip install --upgrade -r $(REQUIREMENTS_DEV)

install-docs-reqs: venv  ## install docs-dependencies
	$(VENV_BIN)/pip install -r $(REQUIREMENTS_DOCS)

upgrade-docs-reqs: venv  ## upgrade docs-dependencies
	@$(VENV_BIN)/pip install --upgrade -r $(REQUIREMENTS_DOCS)

install-pkg-reqs: venv  ## install dependencies
	$(VENV_BIN)/pip install -r $(REQUIREMENTS)

upgrade-pkg-reqs: venv  ## upgrade dependencies
	@$(VENV_BIN)/pip install --upgrade -r $(REQUIREMENTS)

install-reqs: venv  ## install dependencies
	$(VENV_BIN)/pip install -r $(REQUIREMENTS) -r $(REQUIREMENTS_DEV) -r $(REQUIREMENTS_DOCS)

upgrade-reqs: venv  ## upgrade dependencies
	@$(VENV_BIN)/pip install --upgrade -r $(REQUIREMENTS) -r $(REQUIREMENTS_DEV) -r $(REQUIREMENTS_DOCS)

.PHONY: python-venv
python-venv: venv ## start virtualenv python
	exec $(VENV_BIN)/python

.PHONY: bash zsh fish
bash zsh fish: init-venv
	. $(VENV_BIN)/activate && exec $@



