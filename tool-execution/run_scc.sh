#!/bin/bash

## This script sets up, and runs SCC on an Ubuntu 20.04 or MacOS 11 machine.
###########
## SETUP ##
###########
## checks if the environment variable SCC is set
if [ -d ~/scc ]; then
    echo "scc is already setup on this machine. Set your SCC variable by running:"
    echo "export SCC=$(realpath ~/scc/scc)"
    exit 0
fi

SCC_DIR=../scc

## create scc dir if not exists
mkdir -p ${SCC_DIR}

## navigate to scc directory
cd ${SCC_DIR}

# Check OS version
OS=$(uname -a | grep -o '\w*' | head -1)

if [ "$OS" == "Darwin" ]; then
    echo "MacOS detected"
    echo "Downloading SCC for MacOS..."
    wget https://github.com/boyter/scc/releases/download/v3.1.0/scc_3.1.0_Darwin_arm64.tar.gz
elif [ "$OS" == "Linux" ]; then
    echo "Ubuntu detected"
    echo "Downloading SCC for Ubuntu..."
    wget https://github.com/boyter/scc/releases/download/v3.1.0/scc_3.1.0_Linux_arm64.tar.gz
else
    echo "OS not supported"
    exit 1
fi  

## extract scc, move to scc directory
tar -xvf scc_3.1.0_*.tar.gz -C ${SCC_DIR}

## remove the tar file
rm scc_3.1.0_*.tar.gz

#############
## RUN SCC ##
#############
./scc --by-file -f csv -o ../raw_loc_data.csv '../ml_model/src/main/resources/raw_snippet_splitter_out'
./scc --by-file -f csv -o ../loc_data.csv '../ml_model/src/main/resources/snippet_splitter_out'