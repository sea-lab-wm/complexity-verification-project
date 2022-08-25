#!/bin/bash

echo "Running the Typestate Checker. This may take some time..."

java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker simple-datasets/src/main/java/cog_complexity_validation_datasets/One/*.java -d classes -Awarns -Xmaxwarns 10000 2> data/typestate_checker_output_cog_dataset_1.txt

java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker simple-datasets/src/main/java/cog_complexity_validation_datasets/Three/*.java -d classes -Awarns -Xmaxwarns 10000 2> data/typestate_checker_output_cog_dataset_3.txt

# the classpath and supporting variables for dataset 6
DATASET6CP=$(./gradlew :dataset6:printClasspath -q)
# these two variables are not really necessary, but makes the command line a little easier to read
CF314PATH=$(realpath checker-framework-3.14.0/checker/dist/checker.jar)
OUTFILE6=$(realpath data/typestate_checker_output_cog_dataset_6.txt)

java -jar ${CF314PATH} -classpath jatyc.jar:"${DATASET6CP}" -processor jatyc.JavaTypestateChecker -d classes -Awarns -Xmaxwarns 10000 $(find dataset6/src/main/java -name "*.java") 2> ${OUTFILE6}

# the classpath for dataset 6
DATASET9CP=$(./gradlew :dataset9:printClasspath -q)

java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar:"${DATASET9CP}" -processor jatyc.JavaTypestateChecker dataset9/src/main/java/*.java -d classes -Awarns -Xmaxwarns 10000 2> data/typestate_checker_output_cog_dataset_9.txt

java -jar checker-framework-3.14.0/checker/dist/checker.jar -classpath jatyc.jar -processor jatyc.JavaTypestateChecker simple-datasets/src/main/java/fMRI_Study_Classes/*.java -d classes -Awarns -Xmaxwarns 10000 2> data/typestate_checker_output_fMRI_dataset.txt

echo "Typestate Checker successfuly ran. View output in data/typestate_checker_output_{dataset name}.txt"