# from utils import configs
####### Uncomment this to use config for DS3 ##
# from utils import configs_DS3 as configs
from utils import configs_TASK2_DS3 as configs
import json

## read jsonl file
with open(configs.ROOT_PATH + "/" + configs.OUTPUT_PATH) as jsonl_file:
    experiments = [json.loads(jline) for jline in jsonl_file.read().splitlines()]


filtered_experiments_json = []

for exp in experiments:
    if exp["feature_selection_method"] == "Kendalls" and exp["use_oversampling"] == True:
        filtered_experiments_json.append(exp)

## write filtered experiments to jsonl file
with open(configs.ROOT_PATH + "/" + configs.FILTERED_EXPERIMENTS, "w+") as jsonl_file:
   for exp in filtered_experiments_json:
      jsonl_file.write(json.dumps(exp) + "\n")         