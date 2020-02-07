#!/bin/bash

port=$1

# Argument handling
if [ -z "$1" ]
  then
    printf "\nPort argument required.\n\n"
    exit 1
fi

docker run -d -p "${port}":8080 --name coursera-aml-nlp -v $PWD:/root/coursera akashin/coursera-aml-nlp bash -c "run_notebook"

