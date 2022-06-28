::call java -jar ../../../../../../../../checker-framework-3.14.0/checker/dist/checker.jar -classpath ../../../../../../../../jatyc.jar -processor jatyc.JavaTypestateChecker *.java -Awarns -Xmaxwarns 10000 2> ../../../../../../../../data/typestate_checker_output.txt
::pause

call java -jar ../checker-framework-3.14.0/checker/dist/checker.jar -classpath ../jatyc.jar -processor jatyc.JavaTypestateChecker ../src/main/java/edu/wm/kobifeldman/cog_complexity_validation_datasets/Three/*.java  -d classes -Awarns -Xmaxwarns 10000 2> ../data/typestate_checker_output.txt
pause