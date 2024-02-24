.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys, mkdocs

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

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

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
	rm -fr htmlcov

lint: ## check style with flake8
	flake8 ahead tests

coverage: ## check code coverage quickly with the default Python
	coverage run --source ahead setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate docs		
	pip install black pdoc 
	black ahead/* --line-length=80	
	pdoc -t docs ahead/* --output-dir ahead-docs
	find . -name '__pycache__' -exec rm -fr {} +

servedocs: ## compile the docs watching for change	 	
	pip install black pdoc 
	black ahead/* --line-length=80	
	pdoc -t docs ahead/* 
	find . -name '__pycache__' -exec rm -fr {} +

release: dist ## package and upload a release
	pip install twine --ignore-installed
	python3 -m twine upload --repository pypi dist/* --verbose

dist: clean ## builds source and wheel package
	python3 setup.py sdist
	python3 setup.py bdist_wheel	
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python3 -m pip install .

build-site: docs ## export mkdocs website to a folder		
	cp -rf ahead-docs/* ../../Pro_Website/Techtonique.github.io/ahead_python
	find . -name '__pycache__' -exec rm -fr {} +

run-examples: ## run all examples with one command
	find examples -maxdepth 2 -name "*.py" -exec  python3 {} \;

run-mts: ## run all mts examples with one command
	find examples -maxdepth 2 -name "*mts*.py" -exec  python3 {} \;

run-lazy: ## run all lazy examples with one command
	find examples -maxdepth 2 -name "lazy*.py" -exec  python3 {} \;

run-tests: ## run all the tests with one command
	pip install nose2 coverage	
	nose2 -v --with-coverage
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html