RANDOM_SEED = 42  ## previously 632814324
ROOT_PATH = "/Users/nadeeshan/Desktop/Verification-project/complexity-verification-project/ml_model/src/ml-experiments/TASK1-AbsoluteTargets/Classifical-ML-Models"

OUTPUT_RQ1_PATH = "featureselection/experiments_DS6.jsonl"
# OUTPUT_RQ2_PATH="NewExperiments/featureselection/experiments_RQ2_new.jsonl"

DATA_PATH = "data/final_features_ds6.csv"

DISCREATE_TARGETS = ["AU", "ABU", "PBU", "ABU50", "BD", "BD50"]
CONTINOUS_TARGETS = ["TNPU", "TAU"]

WARNING_FEATURES = [
    "warning_sum",
    "warnings_checker_framework",
    "warnings_infer",
    "warnings_openjml",
    "warnings_typestate_checker",
]
# NOT_USEFUL_FEATURES = [
#     "participant_id",
#     "system_name",
#     "file_name",
#     "snippet_signature",
#     "developer_position",
#     "PE gen",
#     "PE spec (java)",
#     "#identifiers",
#     "#identifiers (min)",
# ]
NOT_USEFUL_FEATURES=["dataset_id","snippet_id","person_id","developer_position","PE gen", "#identifiers", "#identifiers (min)", "#literals",
                     "PE spec (java)","time_to_give_output","correct_output_rating","output_difficulty",
                     "brain_deact_31ant","brain_deact_31post","brain_deact_32","time_to_understand",
                     "readability_level", "binary_understandability","correct_verif_questions","gap_accuracy","readability_level_ba"
                     ,"readability_level_before","time_to_read_complete","perc_correct_output","complexity_level","brain_deact_31","method_name","file"]

DATASET_CONFIGS = [
    {"IsDuplicateRemoved": True, "SMOTE": True},
    {"IsDuplicateRemoved": True, "SMOTE": False},
    {"IsDuplicateRemoved": False, "SMOTE": True},
    {"IsDuplicateRemoved": False, "SMOTE": False},
]

## Headers for the output files
KENDALS_HEADER = {
    "feature": "",
    "target": "",
    "tau": 0,
    "|tau|": 0,
    "p-value": 0,
    "dataset": "",
    "drop_duplicates": "",
}
MI_HEADER = {"feature": "", "target": "", "mi": 0, "dataset": "", "drop_duplicates": ""}

## Output file names
KENDALS_OUTPUT_FILE_NAME = "featureselection/kendals_features_DS6.csv"
MI_OUTPUT_FILE_NAME = "featureselection/mutual_info_DS6.csv"

# KENDALS_OUTPUT_FILE_NAME_RQ2="NewExperiments/featureselection/kendals_features_RQ2_new.csv"
# MI_OUTPUT_FILE_NAME_RQ2="NewExperiments/featureselection/mutual_info_RQ2_new.csv"

FS_ALGOS = ["mutual_info_classif", "kendalltau"]


def dataset_geneator(drop_duplicates, dataframe):
    if drop_duplicates:
        return dataframe.drop_duplicates()
    return dataframe
