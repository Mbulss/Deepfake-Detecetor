#!/bin/bash
# Script to clean up Docker and free space on EC2

echo "Cleaning up Docker..."
docker system prune -af --volumes

echo "Cleaning up old images..."
docker image prune -af

echo "Cleaning up build cache..."
docker builder prune -af

echo "Checking disk space..."
df -h

echo "Done! Try building again."

