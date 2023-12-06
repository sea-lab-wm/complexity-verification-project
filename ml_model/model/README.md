## ML-Experiments

1. Create and activate conda environment
```
conda env create -f environment.yml
conda activate ml_model
```

2. Change ROOT_PATH of classification/main.py and results/compare_results.py to your local path.
```
ROOT_PATH=/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/src/main/model/
```

3. Data - For running these ML experiments we use ml_model/model/data/understandability_with_warnings.csv file. This file contains all the code-related features and outputs generated from the verifiability tools.

4. Run classification/main.py generates raw results. Raw results will contain all the results for each model and for each fold. Here we don't need to specify whether to use SMOTE configuration or not because that configuration will be taken from the experiments.jsonl
```
python3 classification/main.py --output_file raw_results.csv
```

5. Run results/classification/data_analysis.py to generate final results.
```
python3 results/classification/data_analysis.py
```