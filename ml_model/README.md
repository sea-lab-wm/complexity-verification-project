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
4. Add unit tests in either ```FeatureExtractortest.java``` or ```FeatureExtractorTestMultiple.java```
If you only have one code snippet to test your feature, use FeatureExtractortest for unit testing. However, if you have multiple code snippets that require testing in a single execution, write the tests in FeatureExtractorTestMultiple.


create_metric_data.py is responsible for collecting the metric and warning data. It outputs to metric_data.csv

feature_data.csv and metric_data.csv are merged into collective table called ml_table.csv. This is done using join_tables.py

#### Tips to Decide whether To Use or Not to use JavaParser.

1. When to use JavaParser

* It is highly recommended to utilize JavaParser whenever possible due to its convenience and ease of use.

* Please refer to the JavaParser documentation available at  https://javadoc.io/doc/com.github.javaparser/javaparser-core/latest/index.html to determine if it provides the necessary API for the desired code feature.

* If you find an API that matches your needs, go ahead and use JavaParser. Means implement your feature by making changes in the FeatureVisitor class.

* Simply for all features that require deeper analysis based on the program's abstract syntax tree.

Examples of features that can be extracted using JavaParser:

* Number of ifstatements
* Number of loops
* Number of method parametemeters


2. When to use SyntacticFeatureExtractor:

* If the feature can be computed unambiguously using a regular expression or a similarly simple technique.

* For features related to the source code format itself (rather than the underlying program).

Examples of features that can be extracted using SyntacticFeatureExtractor:

* Number of lines of non-comment, non-blank code.
* Maximum line length.
* Maximum indentation level.



