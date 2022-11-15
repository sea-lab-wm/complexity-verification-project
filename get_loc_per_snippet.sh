#! /bin/bash

source complexity_verification_project_venv/bin/activate

./complexity_verification_project_venv/bin/python3 separate_snippets.py

cloc loc_per_snippet --skip-uniqueness --by-file --csv --out=loc_per_snippet/output.csv

./complexity_verification_project_venv/bin/python3 loc_dataset_info.py