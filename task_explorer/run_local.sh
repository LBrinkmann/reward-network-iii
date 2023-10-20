#!/bin/bash
set -e -x

if [ ! -d ".venv" ]
then
    python3 -m venv .venv
fi

. .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt

streamlit run app/app.py --server.port=5040 --server.address=0.0.0.0
