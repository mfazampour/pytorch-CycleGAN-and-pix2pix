FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

COPY requirements.txt .

RUN apt update
RUN apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev git
RUN pip install -r requirements.txt


