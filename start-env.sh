#!/bin/bash

name=$1
port=$2

# Argument handling
if [ -z "$1" ]
  then
    printf "\nEnvironment name argument required.\n\n"
    exit 1
fi

if [ -z "$2" ]
  then
    printf "\nPort argument required.\n\n"
    exit 1
fi

docker run -d -p "${port}":8080 --name "${name}" -v $PWD:/root/coursera akashin/coursera-aml-nlp bash -c "run_notebook"

