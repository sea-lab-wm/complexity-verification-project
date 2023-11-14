'''
TODO: incomplete code
'''
ROOT_PATH="/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/"


feature_set_1_path="/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/feature_selection/final_features1_bfs.txt"
feature_set_2_path="/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/feature_selection/final_features2_bfs.txt"
feature_set_3_path="/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/feature_selection/final_features3.txt"
feature_set_4_path="/Users/nadeeshan/Documents/Spring2023/ML-Experiments/complexity-verification-project/ml_model/model/feature_selection/final_features4.txt"


warning_features = ["warnings_checker_framework", "warnings_typestate_checker", "warnings_infer", "warnings_openjml", "warning_sum"]

index = 1
with open(feature_set_1_path) as feature_set_1_file:
    feature_set_1 = feature_set_1_file.read().splitlines()
    for row in feature_set_1:
            target = ''.join(row.split("[")[0])
            features_list = list(''.join(row.split("[")[1]))
            for warning in warning_features:
                features_list.insert(0, warning)
                ## write to experiments.jsonl file
                with open(ROOT_PATH + "classification/experimentsTest.jsonl", "a") as file:
                    st = "{\"experiment_id\": \"exp" + str(index) + "\" ," + "\"target\": \"" + target + "\", \"features\": " + str(features_list) + "\"use_SMOTE\": false }"
                    file.write(st + "\n")
                    st = "{\"experiment_id\": \"exp" + str(index) + "\" ," + "\"target\": \"" + target + "\", \"features\": " + str(features_list) + "\"use_SMOTE\": true }"
                    index += 1