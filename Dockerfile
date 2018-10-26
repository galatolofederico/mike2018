FROM nvidia/cuda:9.0-base

RUN apt-get update && apt-get install -y python3 python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /mike2018

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ADD mnist ./mnist
COPY xor.py .

CMD ["bash"]