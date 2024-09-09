FROM ubuntu:22.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip vim sudo

ENV PYTHONBUFFERED 1

RUN mkdir -p /home/intelligent_intruder
RUN mkdir -p /home/intelligent_intruder/datasets
WORKDIR /home/intelligent_intruder/

COPY requirements_2.txt /home/intelligent_intruder/
RUN pip install --no-cache-dir -r requirements_2.txt

COPY main.py /home/intelligent_intruder/
COPY machine_learning.py /home/intelligent_intruder/
COPY run_main.sh /home/intelligent_intruder/

CMD ["bash"]