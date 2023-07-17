#!/bin/bash

# download docker file
wget https://github.com/milvus-io/milvus/releases/download/v2.2.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# start the docker container
sudo docker-compose up -d

# validate the jobs
sudo docker-compose ps

# install the correct pymilvus version
pip install pymilvus==2.2.0

# clear milvus data
# sudo rm -rf  volumes

# shutdown instance
# sudo docker-compose down
