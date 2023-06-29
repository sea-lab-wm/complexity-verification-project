## Measuring code complexity via ease of verification - Machine Learning based Approach

Build the project                                                                                                          ```./gradlew build```

## Feature Extraction
Code features are extracted by two ways.

* FeatureVisitor.java computes features using the JavaParser. (Example feature: #IfStmts)
* The SyntacticFeatureExtractor.java computes features that easy to compute wtih REGEX than using the JavaParser. (Example feature: #parentheses) 

Both output the collected features to feature_data.csv

### How to Add New Features

1. Add the new feature to the ```Features.java```
2. If the new code feature require JavaParser, implement it in the ```FeatureVisitor.java``` else implement it inside the  ```SyntacticFeatureExtractor.java```
3. Change ```Parser.java``` to add the new feature in the ```features_data.csv```
4. Add unit tests in either ```FeatureExtractortest.java``` (without using DirExplorer) or ```FeatureExtractorTestMultiple.java``` (with using DirExplorer)


create_metric_data.py is responsible for collecting the metric and warning data. It outputs to metric_data.csv

feature_data.csv and metric_data.csv are merged into collective table called ml_table.csv. This is done using join_tables.py