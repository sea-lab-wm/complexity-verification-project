#!/bin/bash

# This script runs OpenJML on the snippets.

if [ "${OJ}x" == "x" ]; then
    echo "run the script tool-execution/openjml-setup-linux.sh if you are on an Ubuntu 20.04 machine. Otherwise, read the instructions in that script on how to adapt it for other platforms, then follow them for yours."
    exit 2
fi

./gradlew clean

HOME="/home/kgdesilva/Desktop/TOSEM/complexity-verification-project/"

# run the fmri dataset. This one is pretty quick and can be run on a laptop
cd ${HOME}simple-datasets/src/main/java && \
    "${OJ}" --esc -Xmaxerrs 10000 $(find fMRI_Study_Classes -name "*.java") &> ../../../../data/openjml_output_fMRI_dataset.txt
echo "Finished for FMRI dataset"

# run dataset 1. This one is fairly fast, too, and can be run on a laptop.
cd ${HOME}simple-datasets/src/main/java && \
    "${OJ}" --esc -Xmaxerrs 10000 $(find cog_complexity_validation_datasets/One/ -name "*.java") &> ../../../../data/openjml_output_cog_dataset_1.txt
echo "Finished for DS1 dataset"

# run dataset 3. This one is slow enough that you shouldn't try to run it on a laptop.
cd ${HOME}simple-datasets/src/main/java && \
    "${OJ}" --timeout 3600 --esc -Xmaxerrs 10000 $(find cog_complexity_validation_datasets/Three/ -name "*.java") &> ../../../../data/openjml_output_cog_dataset_3.txt
echo "Finished for DS3 dataset"

cd ${HOME}
# the classpath for dataset 6
DATASET6CP=$(./gradlew :dataset6:printClasspath -q)

# run dataset 6. This one is slow enough that you shouldn't try to run it on a laptop.
cd ${HOME}dataset6/src/main/java && \
    "${OJ}" --timeout 3600 --esc -cp "${DATASET6CP}" -Xmaxerrs 10000 $(find . -name "*.java") &> ../../../../data/openjml_output_cog_dataset_6.txt
echo "Finished for DS6 dataset"


# dataset 63
cd ..
cd ${HOME}
# the classpath for dataset 63
DATASET63=$(./gradlew :dataset63:printClasspath -q)
cd ${HOME}dataset63/src/main/java && \
    "${OJ}" --timeout 3600 --esc -cp "${DATASET63}" -Xmaxerrs 10000 $(find . -name "*.java") &> ../../../../data/openjml_output_dataset_63.txt
echo "Finished for DS63 dataset"

# dataset 8
cd ..
cd ${HOME}
# the classpath for dataset 8
DATASET8=$(./gradlew :dataset8:printClasspath -q)
cd ${HOME}dataset8/src/main/java && \
    "${OJ}" --timeout 3600 --esc -cp "${DATASET8}" -Xmaxerrs 10000 $(find . -name "*.java") &> ../../../../data/openjml_output_dataset_8.txt
echo "Finished for DS8 dataset"


cd ..
cd ${HOME}

# the classpath for dataset 9
DATASET9CP=$(./gradlew :dataset9:printClasspath -q)

# run dataset 9. This one is slow enough that you shouldn't try to run it on a laptop.
cd ${HOME}dataset9/src/main/java && \
    "${OJ}" --esc -cp "${DATASET9CP}" -Xmaxerrs 10000 $(find . -name "*.java") &> ../../../../data/openjml_output_cog_dataset_9.txt
echo "Finished for DS9 dataset"