#!/bin/bash

dir=$(dirname $0)
source ${dir}/config.sh

nvidia-docker run \
    -d \
    --rm \
    --net=host \
    --env="DISPLAY" \
    -v $HOME:$HOME:rw \
    -v /mnt:/mnt:ro \
    -w=$HOME \
    ${docker_image_name} \
    /opt/pycharm-community-2019.1.2/bin/pycharm.sh
