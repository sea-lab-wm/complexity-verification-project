# from utils import configs
####### Uncomment this to use config for DS3 ##
from utils import configs_DS3 as configs
####### 
import json

## read jsonl file
with open(configs.ROOT_PATH + "/featureselection/experiments_DS3.jsonl") as jsonl_file:
    experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]

## filter experiments
# "feature_selection_method": "Kendalls", 
# "use_oversampling": true
# "warning_features": ["warning_sum"], 
# "drop_duplicates": true,    

filtered_experiments_json = []

for exp in experiments:
    if exp["feature_selection_method"] == "Kendalls" and exp["use_oversampling"] == True and exp["warning_features"] == ["warning_sum"] and exp["drop_duplicates"] == True:
        filtered_experiments_json.append(exp)

## write filtered experiments to jsonl file
with open(configs.ROOT_PATH + "/featureselection/experiments_DS3_filtered.jsonl", "w+") as jsonl_file:
   for exp in filtered_experiments_json:
      jsonl_file.write(json.dumps(exp) + "\n")         