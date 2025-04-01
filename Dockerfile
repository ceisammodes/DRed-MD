FROM ubuntu:latest

LABEL maintainer="Morgane Vacher <morgane.vacher@univ-nantes.fr>"

RUN apt update --snapshot 20250127T000000Z

RUN apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /env
RUN /env/bin/pip install --upgrade pip

ENV PATH="/env/bin:$PATH"

RUN mkdir /dred-md/
COPY requirements.txt /dred-md/

RUN /env/bin/pip install -r /dred-md/requirements.txt

WORKDIR /dred-md/
