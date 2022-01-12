FROM tensorflow/tensorflow:latest-jupyter

MAINTAINER Lucas David <lucasolivdavid@gmail.com>

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

ADD requirements.txt .
RUN pip -qq install -r requirements.txt
ADD notebooks/requirements.txt .
RUN pip -qq install -r requirements.txt
RUN jt -t grade3

ENV JOBLIB_TEMP_FOLDER /data/joblib/
