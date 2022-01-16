#!/bin/bash

sudo docker run -v $PWD:/mnt -it miseyu/docker-ubuntu16-python3.6:latest

# Commands to run in the container:

# cd /mnt
# pip install --upgrade pip
# python3.6 -m pip install -r requirements.txt
