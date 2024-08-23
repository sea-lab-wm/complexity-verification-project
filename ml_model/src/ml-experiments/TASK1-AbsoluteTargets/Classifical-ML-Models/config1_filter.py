from utils import configs
import json

## read jsonl file
with open(configs.ROOT_PATH + "/NewExperiments/featureselection/experiments_RQ1_new.jsonl") as jsonl_file:
    experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]

## filter experiments
# "feature_selection_method": "MI", 
# "use_oversampling": true
# "warning_features": ["warning_sum"], 
# "drop_duplicates": true,    

filtered_experiments_json = []

for exp in experiments:
    if exp["feature_selection_method"] == "MI" and exp["use_oversampling"] == True and exp["warning_features"] == ["warning_sum"] and exp["drop_duplicates"] == True:
        filtered_experiments_json.append(exp)

## write filtered experiments to jsonl file
with open(configs.ROOT_PATH + "/NewExperiments/featureselection/experiments_RQ1_new_filtered.jsonl", "w") as jsonl_file:
   for exp in filtered_experiments_json:
      jsonl_file.write(json.dumps(exp) + "\n")         