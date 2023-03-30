FROM ubuntu:latest
RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/
RUN apt-get update
RUN apt-get install -y python3 && apt-get install -y python3-pip
RUN pip3 install -r requirements.txt
ADD . /app/
EXPOSE 8000
CMD ["gunicorn","--bind=0.0.0.0:80","app:server"]
