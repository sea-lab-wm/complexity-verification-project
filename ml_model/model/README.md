## Machine Learning experiments - Will verification warnings be useful to improve code comprehension?

## Environment Setup

1. Build the docker image locally. Execute below command in the ```<root_directory>/docker``` folder.
```
docker build --build-arg ENV_NAME=ml_model -t ml_model .
```

2. After building the docker image, run the docker container. Execute below command in the ```root``` folder.
```
sh launch_container.sh ml_model
```

### Run Classification Experiments

#### Select Features
1. Run 
```
python3 feature_selection/classification/feature_selection.py
``` 
to generate feature selection results.

2. Run 
```
python3 feature_selection/classification/experiment_generator.py
``` 
to generate experiments and write them into ```classification/experiments.jsonl```

#### Data sets
For Feature Set 1, 2 and 4, for each target there will be separate data files. These are cleaned up removing class overalpping instances and duplicates. For Feature Set 3, there will be a single data file for each target. Different of these data files compared to Feature Set 1 and 2 is that, we are not adding 3 developer features to to remove class overlapping instaces. Instead we removed those instances from the data set.

#### Run Experiments
Run below command to generate raw results.
```
python3 classification/main.py --output_file Final_classification.csv
```

#### Analyse results
1. To plot the histograms of the selected features execute below command.
```
python3 feature_selection/classification/correlation_analysis.py
``` 

2. Run results/compare_results.py to generate final results.
```
python3 Results/classification/data_analysis.py
```

### Run Regression Experiments

#### Select Features
1. Run 
```
python3 feature_selection/regression/feature_selection.py
``` 
to generate feature selection results.

2. Run 
```
python3 feature_selection/regression/experiment_generator.py
``` 
to generate experiments and write them into ```regression/experiments.jsonl```

#### Data sets


#### Run Experiments
Run below command to generate raw results.
```
python3 regression/main.py --output_file Final_regression.csv
```

#### Analyse results
1. To plot the histograms of the selected features execute below command.
```
python3 feature_selection/regression/correlation_analysis.py
``` 

2. Run results/compare_results.py to generate final results.
```
python3 Results/regression/data_analysis.py
```