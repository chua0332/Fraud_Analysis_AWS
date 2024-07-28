#!/bin/bash
repository_uri=791934322985.dkr.ecr.ap-southeast-1.amazonaws.com/fraud_mlops
#aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 791934322985.dkr.ecr.ap-southeast-1.amazonaws.com
# Authenticate Docker to your ECR repository
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin $repository_uri

# Build your Docker image
docker build -t fraud_mlops .

docker buildx create --use
docker buildx build --platform linux/amd64 -t fraud_mlops --load .


# Tag your Docker image
docker tag fraud_mlops:latest $repository_uri:latest

# Push the image to ECR
docker push $repository_uri:latest
