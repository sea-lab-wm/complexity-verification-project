RANDOM_SEED=42  ## previously 632814324
ROOT_PATH="/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/src/ml-experiments/TASK1-AbsoluteTargets/Classifical-ML-Models"
RELATIVE_COMPREHENSIBILITY_ROOT_PATH="/home/nadeeshan/ML-Experiments-2/complexity-verification-project/ml_model/model/NewExperiments/relativecomprehensibility"


OUTPUT_PATH="featureselection/experiments_DS3.jsonl"
OUTPUT_PATH_RELATIVE_COMPREHENSIBILITY="featureselection/experiments_DS3_relative_comprehensibility.jsonl"

DATA_PATH="data/final_features_ds3.csv"
RELATIVE_COMPREHENSIBILITY_DATA_PATH="data/merged_ds3.csv"

TARGET=["(s2>s1)relative_comprehensibility"]

DISCREATE_TARGETS=["readability_level"]
CODE_COMPREHENSIBILITY_TARGETS=["readability_level"]

WARNING_FEATURES = [
    "warning_sum",
    "warnings_checker_framework",
    "warnings_infer",
    "warnings_openjml",
    "warnings_typestate_checker"
]
NOT_USEFUL_FEATURES=["dataset_id","snippet_id","person_id","developer_position","PE gen", "#identifiers", "#identifiers (min)", "#literals",
                     "PE spec (java)","time_to_give_output","correct_output_rating","output_difficulty",
                     "brain_deact_31ant","brain_deact_31post","brain_deact_32","time_to_understand",
                     "binary_understandability","correct_verif_questions","gap_accuracy","readability_level_ba"
                     ,"readability_level_before","time_to_read_complete","perc_correct_output","complexity_level","brain_deact_31","method_name","file"]


NOT_USEFUL_FEATURES_RELATIVE_COMPREHENSIBILITY=["s1","s2", "s1_comprehensibility","s2_comprehensibility","(s2-s1)diff", "epsilon"]
DYNAMIC_EPSILON=[True, False]

DATASET_CONFIGS = [
    {"use_oversampling": True},
    {"use_oversampling": False}
]

## Headers for the output files
## Headers for the output files
KENDALS_HEADER={'feature':'', 'target': '', 'tau': 0, '|tau|':0 ,'p-value':0, 'dataset':'', 'drop_duplicates': ''}
MI_HEADER={'feature':'', 'target': '', 'mi': 0, 'dataset':'', 'drop_duplicates': ''}

## Headers for the output files
RELATIVE_COMPREHENSIBILITY_KENDALS_HEADER={'feature':'', 'target': '', 'dynamic_epsilon': '', 'tau': 0, '|tau|':0 ,'p-value':0}
RELATIVE_COMPREHENSIBILITY_MI_HEADER={'feature':'', 'target': '', 'dynamic_epsilon': '', 'mi': 0}

## Output file names
KENDALS_OUTPUT_FILE_NAME="featureselection/kendals_features_DS3.csv"
MI_OUTPUT_FILE_NAME="featureselection/mutual_info_DS3.csv"

OUTPUT_RQ1_PATH="featureselection/experiments_DS3.jsonl"

FS_ALGOS = ["mutual_info_classif", "kendalltau"]

def dataset_geneator(drop_duplicates, dataframe):
    if drop_duplicates:
        return dataframe.drop_duplicates()
    return dataframe