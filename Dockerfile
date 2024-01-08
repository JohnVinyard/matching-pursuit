FROM python:3.8.13-slim-bullseye

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    build-essential gcc libsndfile1 

RUN pip install \
    torch==1.13.1 \
    falcon zounds \
    numpy==1.23.5 \
    scipy==1.9.3 \
    gunicorn \
    librosa==0.8.0 \
    soundfile==0.9.0 \
    boto3 \
    requests


COPY . /

COPY model-demo/out /

EXPOSE 8888

ENTRYPOINT ["python", "serve.py"]