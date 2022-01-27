# Workshop Neural Network - Identify the genre of a music :computer:

## Table of Contents

1. [Introduction](#introduction-dart)
2. [Requirements](#requirements-books)
3. [Workshop Content](#workshop-content-bulb)
4. [Contact](#contact-telephone_receiver)

## Introduction :dart:

Artificial intelligence is one of the most exciting topics in computer science today. The following workshop session aims to give you a general understanding of how genetic algorithm works, its benefits and limitations, and how to implement it in your own A.I. projects.
Workshop to discover Neural Network through the subject of identify the genre of a music

## Requirements :books:

The workshop is based on python container to ease the work. You must have Docker :whale: installed in your environment.

Once you arrive to the 2nd step, you will need to execute the following commands:

```bash
> sudo docker run -v $PWD:/mnt -w /mnt -it miseyu/docker-ubuntu16-python3.6:latest # run the container

# Once in the container, you can execute the following commands
> apt-get update && apt-get install -y ffmpeg && pip install --upgrade pip && python3.6 -m pip install -r requirements.txt # install the dependencies
> python3.6 test.py # Run the test
```

If you closed the container, you can re-open it by executing the following command:
```bash
> docker ps -a | grep miseyu/docker-ubuntu16-python3.6:latest | awk '{print $1}' # Get the latest container id
> sudo docker start <container_id> # start the container
> sudo docker exec -it <container_id> bash # enter the container
    > cd mnt # enter the workspace
```

## Workshop Content :bulb:

The workshop is divided into 3 parts:

- **Part 1**: Introduction to neural networks and the basics of the machine learning
- **Part 2**: Developing a neural network
- **Part 3**: Testing and upgrading the neural network

**You will find the content of all these parts in the pdf [here](TODO)**

## Sources :notebook:

Datas from [here](https://github.com/mdeff/fma)

How to extract the data from [here](https://github.com/crowdAI/crowdai-musical-genre-recognition-starter-kit)

Helps us [link](https://navdeepsinghh.medium.com/identifying-the-genre-of-a-song-with-neural-networks-851db89c42f0)

Good other project [link](https://towardsdatascience.com/using-cnns-and-rnns-for-music-genre-recognition-2435fb2ed6af)
