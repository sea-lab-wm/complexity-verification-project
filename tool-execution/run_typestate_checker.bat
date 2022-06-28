call java -jar ../checker-framework-3.14.0/checker/dist/checker.jar -classpath ../jatyc.jar -processor jatyc.JavaTypestateChecker ../src/main/java/edu/wm/kobifeldman/cog_complexity_validation_datasets/Three/*.java  -d classes -Awarns -Xmaxwarns 10000 2> ../data/typestate_checker_output_cog_dataset_3.txt
::TODO: Add more datasets here
pause