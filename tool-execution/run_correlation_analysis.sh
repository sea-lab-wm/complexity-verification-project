#! /bin/bash

# run experiments with timeout handing = max
openjml_handle_types=("max" "remove" "zero")

for str in ${openjml_handle_types[@]}; do
    source complexity_verification_project_venv/bin/activate
    ./complexity_verification_project_venv/bin/python3 warning_parser.py ${str}

    if [ "$str" = "remove" ]; then
        ./complexity_verification_project_venv/bin/python3 correlation.py true
    else
        ./complexity_verification_project_venv/bin/python3 correlation.py false
    fi

    cp -f data/correlation_analysis.xlsx data/correlation_analysis_timeout_${str}.xlsx
    cp -f data/raw_correlation_data.csv data/raw_correlation_data_timeout_${str}.csv
done

echo "Correlation Analysis Complete. View results in data/correlation_analysis.xlsx"

#source complexity_verification_project_venv/bin/activate
#./complexity_verification_project_venv/bin/python3 warning_parser.py $handle_type
#./complexity_verification_project_venv/bin/python3 correlation.py true
#cp data/raw_correlation_data.csv data/raw_correlation_data_timeout_$handle_type.csv
#cp ... xlsx...

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