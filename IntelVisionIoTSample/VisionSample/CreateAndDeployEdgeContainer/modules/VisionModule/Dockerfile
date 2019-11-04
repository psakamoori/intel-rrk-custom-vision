FROM python:3.7

#FROM ortopenvino.azurecr.io/onnxruntime-openvino-myriad:latest

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev \
        libboost-python-dev \
        libgtk2.0-dev \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-pip 

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

ENV PATH=/usr/bin/:/bin/

ARG DEVICE

COPY app  .
COPY model ../model
CMD [ "python3", "-u", "./main.py" ]
