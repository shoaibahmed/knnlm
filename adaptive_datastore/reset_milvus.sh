#!/bin/bash

# shutdown instance
echo "Shutting down milvus server"
sudo docker-compose down

# clear milvus data
echo "Removing milvus volumes"
sudo rm -rf  volumes

# start the docker container
echo "Restarting milvus server"
sudo docker-compose up -d

# validate the jobs
echo "Validating job status"
sudo docker-compose ps
