FROM python:3.12.3

WORKDIR /backend

RUN apt-get update && apt-get install -y wget unzip git default-jre-headless && rm -rf /var/lib/apt/lists/*

# Mallet
RUN mkdir -p /backend/Mallet && \
    wget -O /tmp/mallet.zip https://mallet.cs.umass.edu/dist/mallet-2.0.8.zip && \
    unzip /tmp/mallet.zip -d /tmp && \
    mv /tmp/mallet-2.0.8/* /backend/Mallet/ && \
    rm -rf /tmp/mallet*

# NLPipe
RUN git clone https://github.com/lcalvobartolome/NLPipe.git /backend/NLPipe
RUN pip install --no-cache-dir -e /backend/NLPipe/

COPY ./app/backend/requirements.txt /backend/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/backend/ /backend/
COPY ./src/mind /src/mind
COPY ./app/config /src/config
COPY ./app/data /data
COPY ./config.json /backend/NLPipe/config.json

EXPOSE 5001

CMD ["python3", "main.py"]
