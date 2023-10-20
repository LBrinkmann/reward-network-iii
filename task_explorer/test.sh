#!/bin/bash
set -e -x

. .venv/bin/activate

cd app
python -m pytest -vv -s
