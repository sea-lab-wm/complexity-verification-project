#!/bin/bash

# This script runs OpenJML on the snippets. It skips datasets that require timeouts to terminate.

if [ "${OJ}x" == "x" ]; then
    echo "run the script tool-execution/openjml-setup-linux.sh if you are on an Ubuntu 20.04 machine. Otherwise, read the instructions in that script on how to adapt it for other platforms, then follow them for yours."
    exit 2
fi

./gradlew clean

# run the fmri dataset. This one is pretty quick and can be run on a laptop
 cd simple-datasets/src/main/java && \
    "${OJ}" --esc -Xmaxerrs 10000 $(find fMRI_Study_Classes -name "*.java") &> ../../../../data/openjml_output_fMRI_dataset.txt
cd ../../../../
# run dataset 1. This one is fairly fast, too, and can be run on a laptop.
cd simple-datasets/src/main/java && \
    "${OJ}" --esc -Xmaxerrs 10000 $(find cog_complexity_validation_datasets/One/ -name "*.java") &> ../../../../data/openjml_output_cog_dataset_1.txt

# run dataset 3. This one is slow enough that you shouldn't try to run it on a laptop.
# cd simple-datasets/src/main/java && \
#     "${OJ}" --esc -Xmaxerrs 10000 $(find cog_complexity_validation_datasets/Three/ -name "*.java") &> ../../../../data/openjml_output_cog_dataset_3.txt

# the classpath for dataset 6
# DATASET6CP=$(./gradlew :dataset6:printClasspath -q)

# run dataset 6. This one is slow enough that you shouldn't try to run it on a laptop.
# cd dataset6/src/main/java && \
#    "${OJ}" --esc -cp "${DATASET6CP}" -Xmaxerrs 10000 $(find . -name "*.java") &> ../../../../data/openjml_output_cog_dataset_6.txt
cd ../../../../
# the classpath for dataset 9
DATASET9CP=$(./gradlew :dataset9:printClasspath -q)

# run dataset 9. This one is slow enough that you shouldn't try to run it on a laptop.
cd dataset9/src/main/java && \
    "${OJ}" --esc -cp "${DATASET9CP}" -Xmaxerrs 10000 $(find . -name "*.java") &> ../../../../data/openjml_output_cog_dataset_9.txt
