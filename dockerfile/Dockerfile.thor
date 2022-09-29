FROM registry.corp.ailabs.tw/medical/tmi-thor:base-cudnn8.0-torch1.8.1-vision0.9.1
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install -y openslide-tools ca-certificates
COPY . /thor
RUN pip3 install -r /thor/requirements.txt
RUN cd /thor && python3 setup.py install && pre-commit install
