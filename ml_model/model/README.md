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

3. Data - 

3. Run classification/main.py generate raw results.
```
python3 classification/main.py --output_file raw_results.csv
```

4. Run results/compare_results.py to generate final results. For filter SMOTE change use_SMOTE = True in compare_results.py
```
python3 results/compare_results.py
```