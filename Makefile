.PHONY: test install dev venv clean
.ONESHELL:

VENV=.venv
PYTHON=./$(VENV)/bin/python3.11
PIP_INSTALL=$(PYTHON) -m pip install

test:
	$(PYTHON) -m unittest discover

install: venv
	source $(VENV)/bin/activate
	$(PIP_INSTALL) -U pip setuptools wheel
	$(PIP_INSTALL) .

dev: venv
	source $(VENV)/bin/activate
	$(PIP_INSTALL) -U pip setuptools wheel
	$(PIP_INSTALL) -e .[dev]
	pre-commit install

venv:
	test -d $(VENV) || python3.11 -m venv $(VENV)

clean:
	rm -r $(VENV)
	find -iname "*.pyc" -delete
