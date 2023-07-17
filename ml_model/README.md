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

## Procedure to Solve an Issue

In general, this is the procedure you should follow to solve an issue assigned to you:

1. Fork this repository, if you haven't done so. See this: https://docs.github.com/en/get-started/quickstart/fork-a-repo 
2. Clone your repository and check out the ml_model branch. See this: https://www.git-tower.com/learn/git/commands/git-checkout
3. Create a new branch from the ml_model branch. This new branch will be used for addressing the issue (give it a name like "issue[issue_number]", e.g., issue2343).
4. In your new branch, implement the code to solve the issue.
5. If you need to talk to us or have any questions during your implementation, feel free to contact us on Slack
6. Write unit test cases to test your implementation (via JUnit). See this: https://www.vogella.com/tutorials/JUnit/article.html
7. Commit and push your code to your forked repository, referencing the issue in the commit message
8. Create a pull request to this repository and assign Nadeeshan as the reviewer. Reference the issue in the pull request title or description. See this: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
9. We will review your code and give you feedback on how to improve the code.
10. If the code review requests code changes for you, make those changes in the code (in the same branch of the issue).
11. Assign once again Nadeeshan as the code reviewer. 
12. If the code review gives the green light, we will merge your pull request into this repository and we will close the issue.

## Resources

* Abstract Syntax Tree: https://en.wikipedia.org/wiki/Abstract_syntax_tree
* JavaParser book: shared on Slack
* JavaParser API: https://www.javadoc.io/doc/com.github.javaparser/javaparser-core/latest/index.html
* Regular expressions in Java: https://www.vogella.com/tutorials/JavaRegularExpressions/article.html
* Useful VS code plugins: https://code.visualstudio.com/docs/sourcecontrol/github 
