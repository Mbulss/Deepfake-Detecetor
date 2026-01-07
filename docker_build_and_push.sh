#!/bin/bash
# Script to build and optionally push Docker image to Docker Hub

IMAGE_NAME="deepfake-detector"
DOCKER_USERNAME="mbulsss"
FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}"

echo "Building Docker image..."
docker build -t ${IMAGE_NAME} .

echo "Tagging image for Docker Hub..."
docker tag ${IMAGE_NAME} ${FULL_IMAGE_NAME}:latest

echo "Image built and tagged as: ${FULL_IMAGE_NAME}:latest"
echo ""
echo "To push to Docker Hub, run:"
echo "  docker login"
echo "  docker push ${FULL_IMAGE_NAME}:latest"
echo ""
echo "Or run this script with 'push' argument:"
echo "  ./docker_build_and_push.sh push"

if [ "$1" == "push" ]; then
    echo ""
    echo "Pushing to Docker Hub..."
    docker push ${FULL_IMAGE_NAME}:latest
    echo "Done! Image available at: https://hub.docker.com/r/${DOCKER_USERNAME}/${IMAGE_NAME}"
fi

