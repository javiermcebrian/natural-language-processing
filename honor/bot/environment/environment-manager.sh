#!/bin/bash

image=coursera_nlp_honors
image_serve="${image}_serve"
action=$1
service=$2

# Argument handling
if [ -z "$1" ] || [[ -z "$2" && ("$1" == "start" || "$1" == "stop") ]]
  then
    printf "
\nUsage: ./environment-manager.sh action service?\n
Action must be one of the following: start, info, stop, clean.\n
Service must be one of the following (after start or stop actions): pycharm, notebook.\n\n
"
    exit 1
fi

# Functions
case $action in
  start)
    docker build -t "${image_serve}" -f Dockerfile.serve .
    docker-compose up --build -d "${image}_${service}"
    ;;
  info)
    docker ps | grep "${image}"
    ;;
  stop)
    docker-compose stop "${image}_${service}"
    ;;
  clean)
    docker ps -a | awk '{ print $1,$2 }' | grep "${image}" | awk '{ print $1 }' | xargs -I {} docker rm {}
    ;;
  *)
    printf "\nUnknown action argument.\n\n"
    ;;
esac