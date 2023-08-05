FROM pytorch/pytorch:latest

RUN apt update
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN pip3 install torch torchvision torchaudio
RUN pip3 install torchvision

WORKDIR /usr/app/src

COPY cnn.py ./
COPY dataset.py ./
COPY train.py ./

CMD [ "python3" , "./train.py" ]