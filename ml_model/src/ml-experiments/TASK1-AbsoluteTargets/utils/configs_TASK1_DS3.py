RANDOM_SEED=42  ## previously 632814324
ROOT_PATH="/home/nadeeshan/VPro2/complexity-verification-project/ml_model/src/ml-experiments/TASK1-AbsoluteTargets/Classifical-ML-Models"

## output paths ##
OUTPUT_PATH="featureselection/experiments_DS3.jsonl"
OUTPUT_ML_PATH="results/DS3_TASK1_FINAL_No_Transformation.csv"
KENDALS_OUTPUT_PATH="featureselection/kendals_features_DS3.csv"
FILTERED_EXPERIMENTS="featureselection/experiments_DS3_filtered.jsonl"
MI_OUTPUT_PATH="featureselection/mutual_info_DS3.csv"

## Headers for the output files
KENDALS_HEADER={'feature':'', 'target': '', 'tau': 0, '|tau|':0 ,'p-value':0, 'dataset':'', 'drop_duplicates': ''}
MI_HEADER={'feature':'', 'target': '', 'mi': 0, 'dataset':'', 'drop_duplicates': ''}


## data paths ##
DATA_PATH="data/final_features_ds3.csv"

NOT_USEFUL_FEATURES=["dataset_id","snippet_id","person_id","developer_position","PE gen",  
                     "PE spec (java)","time_to_give_output","correct_output_rating","output_difficulty",
                     "brain_deact_31ant","brain_deact_31post","brain_deact_32","time_to_understand",
                     "binary_understandability","correct_verif_questions","gap_accuracy","readability_level_ba",
                     "readability_level_before","time_to_read_complete","perc_correct_output","complexity_level","brain_deact_31","method_name","file"]

# EXTRA_CODE_FEATURES=["#identifiers", "#identifiers (min)", "#literals", "#assignments", "#commas", "#spaces", "#periods"]

TARGETS=["readability_level"]

WARNING_FEATURES = [
    "warning_sum",
    "warnings_checker_framework",
    "warnings_infer",
    "warnings_openjml",
    "warnings_typestate_checker"
]

def dataset_geneator(drop_duplicates, dataframe):
    if drop_duplicates:
        return dataframe.drop_duplicates()
    return dataframe