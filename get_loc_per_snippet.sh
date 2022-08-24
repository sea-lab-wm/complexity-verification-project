#! /bin/bash

source complexity_verification_project_venv/bin/activate

./complexity_verification_project_venv/bin/python3 separate_snippets.py

cloc loc_per_snippet --by-file --csv --out=loc_per_snippet/output.csv

./complexity_verification_project_venv/bin/python3 average_loc_per_snippet.py