FROM python:3.5-slim
LABEL maintainer="Javier Martinez <javiermcebrian@gmail.com>"

RUN apt update && \
    apt install -y \
        openssh-server \
        htop nano git wget \
        libglib2.0-0 autoconf automake \
        libtool build-essential unzip \
        libarchive-dev vim

RUN pip install bert-serving-server==1.10.0 tensorflow==1.10
RUN wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
RUN unzip uncased_L-12_H-768_A-12.zip

CMD bert-serving-start -model_dir uncased_L-12_H-768_A-12 -num_worker=4