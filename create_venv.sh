#!/bin/bash

virtualenv -p python3 qso_timedelay_venv
source qso_timedelay_venv/bin/activate
pip install -r requirements.txt
deactivate
