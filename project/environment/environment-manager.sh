#!/bin/bash

image=coursera_nlp
action=$1

# Argument handling
if [ -z "$1" ]
  then
    printf "\nAction argument is required.\nUsage: ./environment-manager.sh action\nAction must be one of the following: build, start, info, stop, clean.\n\n"
    exit 1
fi

# Functions
case $action in
  build)
    docker-compose build
    ;;
  start)
    docker-compose up --build -d
    ;;
  info)
    docker ps | grep "${image}"
    ;;
  stop)
    docker-compose stop
    ;;
  clean)
    docker ps -a | awk '{ print $1,$2 }' | grep "${image}" | awk '{ print $1 }' | xargs -I {} docker rm {}
    ;;
  *)
    printf "\nUnknown action argument.\n\n"
    ;;
esac