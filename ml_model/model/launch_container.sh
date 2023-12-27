#!/bin/bash
# This script is used to launch the docker container.


# docker build --build-arg ENV_NAME=myenv -t my-docker-image .

# docker image name
DOCKER_IMAGE_NAME="$1"

docker run \
-p 5678:5678 \
--mount src=$(pwd),target=/verification_project,type=bind \
-it ${DOCKER_IMAGE_NAME} /bin/bash