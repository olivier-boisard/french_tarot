#!/bin/bash

source ./config.sh

echo Build docker image ${docker_image_name}

docker build -t ${docker_image_name} \
    --build-arg UID="$(id -u)" \
    --build-arg GID="$(id -g)" \
    --build-arg USERNAME=$(whoami) \
    .
