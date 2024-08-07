RANDOM_SEED=42  ## previously 632814324
ROOT_PATH="/home/nadeeshan/ML-Experiments-2/complexity-verification-project/ml_model/model"

OUTPUT_RQ1_PATH="NewExperiments/featureselection/experiments_RQ1_new.jsonl"
OUTPUT_RQ2_PATH="NewExperiments/featureselection/experiments_RQ2_new.jsonl"

DATA_PATH="data/understandability_with_warnings.csv"

DISCREATE_TARGETS=["AU", "ABU", "PBU", "ABU50", "BD", "BD50"]
CONTINOUS_TARGETS=["TNPU", "TAU"]

WARNING_FEATURES=["warning_sum", "warnings_checker_framework", "warnings_infer", "warnings_openjml", "warnings_typestate_checker"]
NOT_USEFUL_FEATURES=["participant_id", "system_name", "file_name", "snippet_signature"]

DATASET_CONFIGS = [
    {"IsDuplicateRemoved":True, "SMOTE": True},
    {"IsDuplicateRemoved":True, "SMOTE": False},
    {"IsDuplicateRemoved":False, "SMOTE": True},
    {"IsDuplicateRemoved":False, "SMOTE": False}
]

## Headers for the output files
KENDALS_HEADER={'feature':'', 'target': '', 'tau': 0, '|tau|':0 ,'p-value':0, 'dataset':'', 'drop_duplicates': ''}
MI_HEADER={'feature':'', 'target': '', 'mi': 0, 'dataset':'', 'drop_duplicates': ''}

## Output file names
KENDALS_OUTPUT_FILE_NAME="NewExperiments/featureselection/kendals_features_RQ1_new.csv"
MI_OUTPUT_FILE_NAME="NewExperiments/featureselection/mutual_info_RQ1_new.csv"

KENDALS_OUTPUT_FILE_NAME_RQ2="NewExperiments/featureselection/kendals_features_RQ2_new.csv"
MI_OUTPUT_FILE_NAME_RQ2="NewExperiments/featureselection/mutual_info_RQ2_new.csv"

FS_ALGOS = ["mutual_info_classif", "kendalltau"]

def dataset_geneator(drop_duplicates, dataframe):    
    if drop_duplicates:
        return dataframe.drop_duplicates()
    return dataframe