#!/bin/bash

dir=$(dirname "$0")
# shellcheck disable=SC1090
source "${dir}/config.sh"

# shellcheck disable=SC2154
docker run \
    --runtime=nvidia \
    -d \
    --rm \
    --net=host \
    --env="DISPLAY" \
    -v "$HOME":"$HOME":rw \
    -v /mnt:/mnt:ro \
    -w="$HOME" \
    "${docker_image_name}" \
    /opt/pycharm-community-2019.2.3/bin/pycharm.sh
