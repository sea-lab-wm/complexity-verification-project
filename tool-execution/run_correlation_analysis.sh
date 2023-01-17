#! /bin/bash

#cd ..

source complexity_verification_project_venv/bin/activate
./complexity_verification_project_venv/bin/python3 warning_parser.py
./complexity_verification_project_venv/bin/python3 correlation.py
echo "Correlation Analysis Complete. View results in data/correlation_analysis.xlsx"

# # run experiments with timeout handing = max
# source complexity_verification_project_venv/bin/activate
# ./complexity_verification_project_venv/bin/python3 warning_parser.py max
# ./complexity_verification_project_venv/bin/python3 correlation.py
# cp ../data/raw_data.csv ../data/radata_timeout_max.csv
# cp ... xlsx...

# # run experiments with timeout handing = zero
# handing_type=zero
# source complexity_verification_project_venv/bin/activate
# ./complexity_verification_project_venv/bin/python3 warning_parser.py $handing_type
# ./complexity_verification_project_venv/bin/python3 correlation.py
# cp ../data/raw_data.csv ../data/radata_timeout_$handing_type.csv
# cp ... xlsx...


# # run experiments with timeout handing = remove
# handing_type=remove
# source complexity_verification_project_venv/bin/activate
# ./complexity_verification_project_venv/bin/python3 warning_parser.py $handing_type
# ./complexity_verification_project_venv/bin/python3 correlation.py
# cp ../data/raw_data.csv ../data/radata_timeout_$handing_type.csv
# cp ... xlsx...
# echo "Correlation Analysis Complete. View results in data/correlation_analysis.xlsx"