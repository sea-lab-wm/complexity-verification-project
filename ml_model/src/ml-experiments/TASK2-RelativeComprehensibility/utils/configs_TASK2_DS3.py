RANDOM_SEED=42  ## previously 632814324

ROOT_PATH="/home/nadeeshan/VPro2/complexity-verification-project/ml_model/src/ml-experiments/TASK2-RelativeComprehensibility"
# ROOT_PATH="/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/src/ml-experiments/TASK2-RelativeComprehensibility"

USE_DS3_CONFIGS=True
USE_DS6_CONFIGS=False
USE_MERGED_CONFIGS=False

## Data paths
RAW_DATA_PATH="data/final_features_ds3.csv"
DATA_PATH="data/merged_ds3.csv"

## Not useful features for the feature selection - used when creating the merged dataset (i.e. computing the RC)
NOT_USEFUL_FEATURES=["dataset_id", "person_id","developer_position","PE gen", 
                     "PE spec (java)","time_to_give_output","correct_output_rating","output_difficulty",
                     "brain_deact_31ant","brain_deact_31post","brain_deact_32","time_to_understand",
                     "binary_understandability","correct_verif_questions","gap_accuracy","readability_level_ba",
                     "readability_level_before","time_to_read_complete","perc_correct_output","complexity_level","brain_deact_31","method_name","file"]


## Not useful features for the feature selection
NOT_USEFUL_FEATURES_RC=["s1","s2", "s1_comprehensibility","s2_comprehensibility","(s2-s1)diff", "epsilon"]

## Headers for the output files
ML_OUTPUT_PATH="results/DS3_TASK2_FINAL_Standard.csv"
OUTPUT_PATH="featureselection/experiments_DS3.jsonl"
FILTERED_EXPERIMENTS="featureselection/experiments_DS3_filtered.jsonl"
KENDALS_HEADER={'feature':'', 'target': '', 'dynamic_epsilon': '', 'tau': 0, '|tau|':0 ,'p-value':0, 'drop_duplicates': ''}
MI_HEADER={'feature':'', 'target': '', 'dynamic_epsilon': '', 'mi': 0, 'drop_duplicates': ''}

## Output file names
KENDALS_OUTPUT_FILE_NAME="featureselection/kendals_features_DS3.csv"
MI_OUTPUT_FILE_NAME="featureselection/mutual_info_DS3.csv"

S1_FEATURES_INTERMEDIATE="data/DS3_s1_features_with_warnings.csv"
S2_FEATURES_INTERMEDIATE="data/DS3_s2_features_with_warnings.csv"

RC_EPSILON_STATIC_PATH="data/DS3_relative_comprehensibility_e_0_static.csv"
RC_EPSILON_DYNAMIC_PATH="data/DS3_relative_comprehensibility_e_dynamic.csv"

RC_EPSILON_STATIC_TRAIN_DATA="data/DS3_train_date_with_warnings_e_0_static.csv"
RC_EPSILON_DYNAMIC_TRAIN_DATA="data/DS3_train_date_with_warnings_e_dynamic.csv"

CODE_COMPREHENSIBILITY_TARGETS=["readability_level"]

DYNAMIC_EPSILON=[True, False]

WARNING_FEATURES=["warnings_checker_framework_x","warnings_typestate_checker_x","warnings_infer_x","warnings_openjml_x","warning_sum_x", "warnings_checker_framework_y","warnings_typestate_checker_y","warnings_infer_y","warnings_openjml_y","warning_sum_y"]

TARGET=["(s2>s1)relative_comprehensibility"]

def dataset_geneator(drop_duplicates, dataframe):
    if drop_duplicates:
        return dataframe.drop_duplicates()
    return dataframe