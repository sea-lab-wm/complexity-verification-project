# Measuring the effect of code verifiability on ML-based prediction of comprehensibility

This project contains all the code that extracts code features from a set of Java code snippets. 
These features are used by ML models to predict comprehensibility.

In another project, we computed verifiability features for those snippets, using verification tools (aka verifiers). 
These features are also used by the ML models.

## Requirements and Execution
Requirements: Java 11+ from OpenJDK (recommended) or another vendor

To build and execute the project, execute:
```./gradlew build```

This would produce the file: `feature_data.csv`

## Feature Extraction
Code features are extracted in two ways:

1. ```FeatureVisitor.java``` computes features by parsing the code and traversing its Abstract Syntax Tree (AST) using the JavaParser: https://javaparser.org/. (Example feature: #IfStmts)
2. The ```SyntacticFeatureExtractor.java``` computes features that are easy to compute with regular expressions (regexes) or heuristics, rather than using the JavaParser. (Example feature: #parentheses) -- See this tutorial about Java regular expressions: https://www.vogella.com/tutorials/JavaRegularExpressions/article.html

Both classes are used in the class `Parser` to collect the features in `feature_data.csv`

### How to Implement Extractors of  New Features?

1. Add the new feature to the ```Features.java```
2. If the new code feature requires JavaParser, implement it in the ```FeatureVisitor.java``` else implement it inside the  ```SyntacticFeatureExtractor.java```
3. Change ```Parser.java``` to add the new feature in the ```features_data.csv```
4. Add unit tests in either ```FeatureExtractorTest.java``` or ```FeatureExtractorTestMultiple.java```
If you only have one code snippet to test your feature, use ```FeatureExtractorTest``` for unit testing. However, if you have multiple code snippets in a directory that require testing in a single execution, write the tests in ```FeatureExtractorTestMultiple```.

`create_metric_data.py` is responsible for collecting the metric and warning data from the verifiers. It outputs to `metric_data.csv`

`feature_data.csv` and `metric_data.csv` are merged into a collective table called `ml_table.csv`. This is done using `join_tables.py`

#### Tips to Decide whether To Use or Not to Use the JavaParser.

**1. When to use JavaParser?**

* It is highly recommended to utilize JavaParser whenever possible due to its convenience and ease of use.

* Please refer to the JavaParser documentation available at  https://javadoc.io/doc/com.github.javaparser/javaparser-core/latest/index.html to determine if it provides the necessary API for the desired code feature.

* If you find an API that matches your needs, go ahead and use JavaParser. Implement your feature by making changes in the FeatureVisitor class.

* Simply for all features that require deeper analysis based on the program's abstract syntax tree.

Examples of features that can be extracted using JavaParser:

* Number of ifstatements
* Number of loops
* Number of method parametemeters

**2. When to use SyntacticFeatureExtractor?**

* If the feature can be computed unambiguously using a regular expression or a similarly simple technique.

* For features related to the source code format itself (rather than the underlying program).

Examples of features that can be extracted using SyntacticFeatureExtractor:

* Number of lines of non-comment, non-blank code.
* Maximum line length.
* Maximum indentation level.



