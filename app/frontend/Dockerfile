FROM python:3.12.3

WORKDIR /frontend

COPY ./app/frontend/ . 
COPY app/.env .env 

ENV PYTHONPATH=/frontend

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python3", "main.py"]
