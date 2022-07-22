#! /bin/bash

echo "Running the Typestate Checker. This may take some time..."

java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker simple-datasets/src/main/java/cog_complexity_validation_datasets/One/*.java -d classes -Awarns -Xmaxwarns 10000 2> data/typestate_checker_output_cog_dataset_1.txt
java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker simple-datasets/src/main/java/cog_complexity_validation_datasets/Three/*.java -d classes -Awarns -Xmaxwarns 10000 2> data/typestate_checker_output_cog_dataset_3.txt
java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker dataset6/src/main/java/*.java -d classes -Awarns -Xmaxwarns 10000 2> data/typestate_checker_output_cog_dataset_6.txt
java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker dataset6/src/main/java/weka/estimators/*.java -d classes -Awarns -Xmaxwarns 10000 2>> data/typestate_checker_output_cog_dataset_6.txt
java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker dataset6/src/main/java/weka/gui/beans/*.java -d classes -Awarns -Xmaxwarns 10000 2>> data/typestate_checker_output_cog_dataset_6.txt
java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker dataset6/src/main/java/weka/gui/experiment/*.java -d classes -Awarns -Xmaxwarns 10000 2>> data/typestate_checker_output_cog_dataset_6.txt
java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker dataset9/src/main/java/*.java -d classes -Awarns -Xmaxwarns 10000 2> data/typestate_checker_output_cog_dataset_9.txt
java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker simple-datasets/src/main/java/fMRI_Study_Classes/*.java -d classes -Awarns -Xmaxwarns 10000 2> data/typestate_checker_output_fMRI_dataset.txt

echo "Typestate Checker successfuly ran. View output in data/typestate_checker_output_{dataset name}.txt"