#!/bin/bash

IMAGE_NAME=xtract_bert_image

docker rmi -f $IMAGE_NAME

docker build -t $IMAGE_NAME .
