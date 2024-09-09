RANDOM_SEED=42  ## previously 632814324

ROOT_PATH="/Users/admin/Desktop/complexity-verification-project/ml_model/src/ml-experiments/TASK2-RelativeComprehensibility"

USE_DS3_CONFIGS=False
USE_DS6_CONFIGS=False
USE_MERGED_CONFIGS=True

## Data paths
DATA_PATH="data/combined_ds.csv"

## Not useful features for the feature selection
NOT_USEFUL_FEATURES=["s1","s2", "s1_comprehensibility","s2_comprehensibility","(s2-s1)diff", "epsilon"]

## Headers for the output files
ML_OUTPUT_PATH="results/COMBINED_DS_TASK2.csv"
OUTPUT_PATH="featureselection/experiments_COMBINED.jsonl"
FILTERED_EXPERIMENTS="featureselection/experiments_COMBINED_filtered.jsonl"
KENDALS_HEADER={'feature':'', 'target': '', 'dynamic_epsilon': '', 'tau': 0, '|tau|':0 ,'p-value':0}
MI_HEADER={'feature':'', 'target': '', 'dynamic_epsilon': '', 'mi': 0}

## Output file names
KENDALS_OUTPUT_FILE_NAME="featureselection/kendals_features_COMBINED.csv"
MI_OUTPUT_FILE_NAME="featureselection/mutual_info_COMBINED.csv"

CODE_COMPREHENSIBILITY_TARGETS=["(s2>s1)relative_comprehensibility"]

DYNAMIC_EPSILON=[True, False]

WARNING_FEATURES=["warnings_checker_framework_x","warnings_typestate_checker_x","warnings_infer_x","warnings_openjml_x","warning_sum_x", "warnings_checker_framework_y","warnings_typestate_checker_y","warnings_infer_y","warnings_openjml_y","warning_sum_y"]

TARGET=["(s2>s1)relative_comprehensibility"]