#!/bin/sh

# This script runs the Scalabrino feature extraction tool on a given Java file.



## list the directory names in the corrected_raw_snippets directory and save them in a array
# dirs=($(ls corrected_raw_snippets))

## If need to compute the same snippets used to compute features with our code uncomment this ##
dirs=($(ls corrected_raw_snippets_with_package_class_names))

for dir in ${dirs[@]}
do
    ## list the files in the directory and save them in a array
    # files=($(ls corrected_raw_snippets/${dir}))
    files=($(ls corrected_raw_snippets_with_package_class_names/${dir}))
    for file in ${files[@]}
    do
        ## split the file name to get the dataset_id, snippet_id, method_name
        IFS='_' read -ra meta_data <<< "$file"
        dataset_id=${meta_data[1]}
        ## replace $ in the snippet_id with -
        snippet_id=$(echo ${meta_data[3]} | sed 's/\$/-/g')
        
        method_name=${meta_data[4]}
        IFS='.' read -ra method_name <<< "$method_name"
        method_name=${method_name[0]}

        echo "Processing ${dir}/${file} with dataset_id: ${dataset_id}, snippet_id: ${snippet_id}, method_name: ${method_name}"
        ## run the feature extraction tool to extract the features
        # java -cp rsm.jar it.unimol.readability.metric.runnable.ExtractMetrics corrected_raw_snippets/${dir}/${file} > output/${file}_features.txt  
        java -cp rsm.jar it.unimol.readability.metric.runnable.ExtractMetrics corrected_raw_snippets_with_package_class_names/${dir}/${file} > output_classes_packages/${file}_features.txt  
        
    done
done

## run the feature extraction tool to extract the "readability" feature
# java -jar rsm.jar corrected_raw_snippets/**/*.java > readability_features_DELETE.txt
java -jar rsm.jar corrected_raw_snippets_with_package_class_names/**/*.java > readability_features_classes_packages.txt

