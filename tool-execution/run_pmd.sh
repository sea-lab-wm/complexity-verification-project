#!/bin/bash

## This script sets up, and runs PMD on an Ubuntu 20.04 or MacOS 11 machine.
###########
## SETUP ##
###########
## checks if the environment variable pmd is set
if ! command -v pmd &> /dev/null ; then
    echo "cannot find pmd on this machine."
    echo "run the script tool-execution/pmd-setup.sh if you are on an Ubuntu 20.04 machine or Mac OS 11+. Otherwise, read the instructions in that script on how to adapt it for other platforms, then follow them for yours."
    ## run pmd-setup.sh to setup pmd
    ./pmd_setup.sh

    PMD_DIR=../pmd

    ## create config.xml file
    echo "<?xml version='1.0'?>
<ruleset name='Custom Rule for Cyclomatic Complexity'>
    <description>Cyclomatic complexity</description>
    <rule ref='category/java/design.xml/CyclomaticComplexity'>
        <properties>
            <property name='methodReportLevel' value='1' />
        </properties>
    </rule>
</ruleset>" > ${PMD_DIR}/cyclomatic_complex_rules.xml

    export PMD=$(realpath ${PMD_DIR}/pmd-bin-7.0.0-rc3/bin/pmd)

    #############
    ## RUN PMD ##
    ## ##########
    echo "running pmd on the raw snippets..."
    ${PMD} check -d ../ml_model/src/main/resources/raw_snippet_splitter_out -R ${PMD_DIR}/cyclomatic_complex_rules.xml -f csv -r ../ml_model/raw_cyclomatic_complexity_data.csv

    echo "running pmd on the processed snippets..."
    ${PMD} check -d ../ml_model/src/main/resources/snippet_splitter_out -R ${PMD_DIR}/cyclomatic_complex_rules.xml -f csv -r ../ml_model/cyclomatic_complexity_data.csv

else
    echo "pmd installation found on this machine."
    echo "running pmd on the raw snippets..."
    pmd check -d ../ml_model/src/main/resources/raw_snippet_splitter_out -R ${PMD_DIR}/cyclomatic_complex_rules.xml -f csv -r ../ml_model/raw_cyclomatic_complexity_data.csv
    echo "running pmd on the processed snippets..."
    pmd check -d ../ml_model/src/main/resources/snippet_splitter_out -R ${PMD_DIR}/cyclomatic_complex_rules.xml -f csv -r ../ml_model/cyclomatic_complexity_data.csv
fi