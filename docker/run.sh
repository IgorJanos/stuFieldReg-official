#!/usr/bin/env bash

# Fail on error and unset variables.
set -eu -o pipefail

CWD=$(readlink -e "$(dirname "$0")")
cd "${CWD}/.." || exit $?
source ./docker/common.sh

DEVICE=$1
echo "Using GPU devices: ${DEVICE}"

DEVICE_NAME=`echo ${DEVICE} | tr "," "-"`

export USER_NAME=$(whoami)
echo "User: ${USER_NAME}"


docker run \
    -it --rm \
    --name "stuFieldReg-${DEVICE_NAME}" \
    --gpus all \
    --privileged \
    --shm-size 64g \
    --device /dev/fuse \
    -v "${HOME}/.netrc":/root/.netrc \
    -v "${CWD}/..":/workspace/${PROJECT_NAME} \
    -v "/mnt/scratch/${USER}/.datasets":/mnt/datasets \
    -v "/mnt/scratch/${USER}/${PROJECT_NAME}":/workspace/${PROJECT_NAME}/.scratch \
    -e CUDA_VISIBLE_DEVICES="${DEVICE}" \
    ${IMAGE_TAG} 