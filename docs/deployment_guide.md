# Deployment Guide: ChurnBuster API

This guide provides instructions on how to run and deploy the ChurnBuster prediction API using Docker.

## 1. Prerequisites
- **Docker:** You must have Docker Desktop installed and running on your machine.
- **Project Files:** You need a complete copy of the project repository.

## 2. Building the Docker Image
The `Dockerfile` in the root directory contains all the instructions to build a self-contained image with the application and its dependencies.

To build the image, navigate to the project's root directory (`churnbuster`) in your terminal and run the following command:

```bash
docker build -t churnbuster-api .