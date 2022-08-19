#!/bin/sh

# This script downloads and sets up OpenJML on an Ubuntu 20.04 machine. Run it before running
# run_openjml.sh. If you're on a different platform, follow similar steps, but replace
# the wget command with the one that's appropriate for your platform (but do note that OpenJML
# only supports Ubuntu 20.04, Ubuntu 18.04, MacOS 11, and MacOS 10.15 as of the time of this
# writing).

if [ -d ~/openjml ]; then
   echo "openjml is already setup on this machine. Set your OJ variable by running:"
   echo "export OJ=$(realpath ~/openjml/)"
   exit 0
fi

mkdir ~/openjml
cd ~/openjml
wget https://github.com/OpenJML/OpenJML/releases/download/0.17.0-alpha-15/openjml-ubuntu-20.04-0.17.0-alpha-15.zip
unzip openjml-ubuntu-20.04-0.17.0-alpha-15.zip
export OJ_DIR=$(realpath ~/openjml/)
export OJ=${OJ_DIR}/openjml

# remove bad spec files
rm ${OJ_DIR}/specs/java/awt/geom/Point2D.jml
rm ${OJ_DIR}/specs/java/awt/Color.jml
rm ${OJ_DIR}/specs/java/awt/event/ActionListener.jml
