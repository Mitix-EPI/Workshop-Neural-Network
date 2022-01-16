#!/bin/bash

sudo docker run -v $PWD:/mnt -it miseyu/docker-ubuntu16-python3.6:latest

# If you quit the container, you can run the container again by typing:
# docker start <container_id>
# docker exec -it <container_id> bash
# don't forget to stop the container by typing:
# docker stop <container_id>

# Commands to run in the container:

# cd /mnt
# apt-get update && apt-get install -y ffmpeg
# pip install --upgrade pip
# python3.6 -m pip install -r requirements.txt
# python3.6 download_songs.py # if you want to download songs
# python3.6 convert.py # YOU MUST RUN THIS SCRIPT (it parses the song's metadatas files)
