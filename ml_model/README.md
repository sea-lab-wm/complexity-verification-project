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

## Split method into code snippets

**Note: Code splitting should be performed only once!** 

### Standardize Parsing + Formatting 
If the snippet_spitter_out already contains the snippets (including the ones mentioned in Step 2), there is no need to proceed with the following steps. 

1. Create directories ```ml_model/src/main/resources/manually_created_snippets/ds_3``` and ```ml_model/src/main/resources/manually_created_snippets/ds_6``` 
  
2. Copy below manually created snippets into the above directories accordingly. ds_3 ones to ds_3 folder and ds_6 ones to ds_6 folder.
snippets: ```ds_3_snip_35_DisbandUnitAction```,```ds_3_snip_43_TestClassRunnerForParameters```, ```ds_3_snip_48_ComparisonFailure```, ```ds_6_snip_1$Pom_HealthReport```, ```ds_6_snip_2$HibernateORM_TimesTenDialect```, ```ds_6_snip_4$K9_StorageManager```

3. Run ```edu.wm.sealab.featureextraction.SnippetSplitter.main()```

4. In order to format the splitted snippets, use spotless like below in the ml_model/build.gradle
```
spotless {
   java {
     importOrder()
     removeUnusedImports()
     googleJavaFormat("1.8")
     target 'src/main/resources/snippet_splitter_out/*.java'
   }
 }
```

### Raw snippets
1. Follow the steps (1,2) in the section above.

2. Replace line 175 in the SnippetSplitter.java
`String methodString = method.toString();`  
with
`String methodString = LexicalPreservingPrinter.print(method);`

3. Run ```edu.wm.sealab.featureextraction.SnippetSplitter.main()```




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

Current Status of the Feature Computation:

|  | Code Feature  | Flavours |Description | Implementation |
| -- | ------------- | -- | ------------- | -- |
|1|Cyclomatic comp | - [x] Non aggregated | It is computed the program complexity using the Control Flow Graph of the program. | [PMD tool](https://github.com/pmd/pmd) |
|2|#nested blocks | - [x] Avg (divided by code lines) | counts all the child nodes inside a block statement. Types of child nodes => block_stmt, for_stmt, foreach_stmt, while_stmt, do_stmt, if_stmt, if_else_stmt, switch_stmt, try_stmt, catch_stmt, synchronized_stmt  | AST (Java Parser) |
|3|#parameters
|4|#statements
|5|#assignments
|6|#blank lines
|7|#characters
|8|#commas
|9|#comments
|10|#comparisons
|11|#conditionals
|12|#identifiers
|13|#keywords
|14|#literals
|15|#loops
|16|#numbers
|17|#operators
|18|#parenthesis
|19|#periods
|20|#spaces
|21|#strings
|22|#words
|23|Indentation length
|24|Identifiers length
|26|Line length
|27|#aligned blocks
|28|Extent of aligned blocks
|29|Entropy
|30|LOC
|31|Volume
|32|NMI (Narrow Meaning identifier)
|33|NM
|34|ITID
|35|TC
|36|Readability
|37|IMSQ





## Procedure to Solve an Issue

In general, this is the procedure you should follow to solve an issue assigned to you:

1. Fork this repository, if you haven't done so. See this: https://docs.github.com/en/get-started/quickstart/fork-a-repo 
2. Clone your repository and check out the `ml_model` branch. See this: https://www.git-tower.com/learn/git/commands/git-checkout
3. Create a new branch from the `ml_model` branch. This new branch will be used for addressing the issue (give it a name like "issue[issue_number]", e.g., `issue2343`).
4. In your new branch, implement the code to extract one feature or a small subset of features (specified in the issue you are solving). 
5. If you need to talk to us or have any questions during your implementation, feel free to contact us on Slack
6. Write unit test cases to test your implementation (via JUnit). See this: https://www.vogella.com/tutorials/JUnit/article.html
7. Commit and push your code to your forked repository, referencing the issue in the commit message (see this: https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls#issues-and-pull-requests).
8. Create a pull request to this repository and assign Nadeeshan as the reviewer. Reference the issue in the pull request title or description. See this: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
9. We will review your code and give you feedback on improving your implementation/tests.
10. If the review requests code changes for you, make those changes in the code (in the same branch of the issue). Pushing those changes will automatically update the pull request.
11. Assign Nadeeshan once again as the code reviewer. 
12. If the code review gives the green light, we will merge your pull request into this repository and we will close the issue.

## Resources

* Abstract Syntax Tree: https://en.wikipedia.org/wiki/Abstract_syntax_tree
* JavaParser book: https://drive.google.com/file/d/14EqlTrh61vYfkTEU9FU6As0B7MvrZADq/view?usp=sharing
* JavaParser API: https://www.javadoc.io/doc/com.github.javaparser/javaparser-core/latest/index.html
* Regular expressions in Java: https://www.vogella.com/tutorials/JavaRegularExpressions/article.html
* Useful VS code plugins: https://code.visualstudio.com/docs/sourcecontrol/github 
