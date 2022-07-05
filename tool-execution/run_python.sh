#! /bin/bash

cd ..
source complexity_verification_project_venv/bin/activate
./complexity_verification_project_venv/bin/python3 parser.py
./complexity_verification_project_venv/bin/python3 correlation.py
echo "Correlation Analysis Complete. View results in data/correlation_analysis.xlsx"