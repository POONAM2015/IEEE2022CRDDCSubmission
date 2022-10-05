FROM nvcr.io/nvidia/pytorch:21.08-py3

# apt install required packages
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y wget zip htop screen libgl1-mesa-glx git
# pip install required packages
RUN pip install seaborn thop

WORKDIR /usr/src/app/
RUN git clone https://github.com/WongKinYiu/yolov7.git
WORKDIR /usr/src/app/yolov7/

ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt /usr/src/app/yolov7/
ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt /usr/src/app/yolov7/
ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt /usr/src/app/yolov7/
ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt /usr/src/app/yolov7/
ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt /usr/src/app/yolov7/
ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt /usr/src/app/yolov7/

ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt /usr/src/app/yolov7/
ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x_training.pt /usr/src/app/yolov7/
ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6_training.pt /usr/src/app/yolov7/
ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6_training.pt /usr/src/app/yolov7/
ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6_training.pt /usr/src/app/yolov7/
ADD https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e_training.pt /usr/src/app/yolov7/
