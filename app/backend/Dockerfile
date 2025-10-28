FROM python:3.12.3

RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre-headless && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /backend

COPY ./app/backend/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/backend/ .
RUN pip install --no-cache-dir -e ./NLPipe/
COPY ./src/mind /src/mind
COPY ./app/config /src/config
COPY ./app/.env . 
COPY ./app/data /data

ENV PYTHONPATH=/backend:/src

EXPOSE 5001

CMD ["python3", "main.py"] 
