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