RANDOM_SEED=42  ## previously 632814324
ROOT_PATH="/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/src/ml-experiments/TASK2-RelativeComprehensibility/"
RELATIVE_COMPREHENSIBILITY_ROOT_PATH="/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/src/ml-experiments/TASK2-RelativeComprehensibility"


OUTPUT_RQ1_PATH="featureselection/experiments_DS6.jsonl"
ORIGINAL_DATA_WITH_WARNINGS="data/final_features_ds6.csv"

DATA_PATH="data/merged_ds6.csv"

TARGET=["(s2>s1)relative_comprehensibility"]
NOT_USEFUL_FEATURES=["dataset_id", "person_id","developer_position","PE gen", "#identifiers", "#identifiers (min)", "#literals",
                     "PE spec (java)","time_to_give_output","correct_output_rating","output_difficulty",
                     "brain_deact_31ant","brain_deact_31post","brain_deact_32","time_to_understand",
                     "readability_level", "binary_understandability","correct_verif_questions","gap_accuracy","readability_level_ba"
                     ,"readability_level_before","time_to_read_complete","perc_correct_output","complexity_level","brain_deact_31","method_name","file"]

CODE_COMPREHENSIBILITY_TARGETS=["ABU", "AU", "ABU50", "BD", "BD50", "PBU"]

WARNING_FEATURES=["warnings_checker_framework_x","warnings_typestate_checker_x","warnings_infer_x","warnings_openjml_x","warning_sum_x", "warnings_checker_framework_y","warnings_typestate_checker_y","warnings_infer_y","warnings_openjml_y","warning_sum_y"]
NOT_USEFUL_FEATURES_RC=["s1","s2", "s1_comprehensibility","s2_comprehensibility","(s2-s1)diff", "epsilon"]

DYNAMIC_EPSILON=[True, False]
DATASET_CONFIGS = [
    {"use_oversampling": True},
    {"use_oversampling": False}
]

## Headers for the output files
KENDALS_HEADER={'feature':'', 'target': '', 'dynamic_epsilon': '', 'tau': 0, '|tau|':0 ,'p-value':0}
MI_HEADER={'feature':'', 'target': '', 'dynamic_epsilon': '', 'mi': 0}

## Output file names
KENDALS_OUTPUT_FILE_NAME="featureselection/kendals_features_DS6.csv"
MI_OUTPUT_FILE_NAME="featureselection/mutual_info_DS6.csv"

FS_ALGOS = ["mutual_info_classif", "kendalltau"]

def dataset_geneator(drop_duplicates, dataframe):    
    if drop_duplicates:
        return dataframe.drop_duplicates()
    return dataframe