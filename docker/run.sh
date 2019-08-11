#!/bin/bash

dir=$(dirname $0)
source ${dir}/config.sh

docker run \
    --runtime=nvidia \
    -d \
    --rm \
    --net=host \
    --env="DISPLAY" \
    -v $HOME:$HOME:rw \
    -v /mnt:/mnt:ro \
    -w=$HOME \
    ${docker_image_name} \
    /opt/pycharm-community-2019.2/bin/pycharm.sh
