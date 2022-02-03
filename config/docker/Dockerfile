ARG BASE_IMAGE
FROM $BASE_IMAGE

MAINTAINER Lucas David <lucasolivdavid@gmail.com>

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

ADD requirements.txt .
RUN pip -qq install -r requirements.txt
ADD notebooks/requirements.txt .
RUN pip -qq install -r requirements.txt
RUN jt -t grade3
