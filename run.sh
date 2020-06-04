#!/bin/bash

IMAGE_NAME=xtract_bert_image

args_array=("$@")
DIRECTORY=("${args_array[@]:0:1}")

echo "docker run -it -v $DIRECTORY:/$DIRECTORY $IMAGE_NAME --path /$DIRECTORY
"

docker run -it -v $DIRECTORY:/$DIRECTORY $IMAGE_NAME --path /$DIRECTORY
