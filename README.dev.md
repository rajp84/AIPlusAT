# Local development setup

## Start up
Start up all the required components

`docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml -f docker-compose.local-override.yml up -d`

This should start everything _except_ `cvat_server` and `cvat_ui`

## Build/Start Server

Build a docker image of the server

`docker build -f Dockerfile -t cvat/server:local .`

Start the `cvat_server` docker contianer

`CVAT_VERSION=local docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.local-override.yml up -d cvat_server`

## Run UI Locally

Start the UI locally

`yarn workspace cvat-ui run start`

UI is available at `http://localhost:8080`