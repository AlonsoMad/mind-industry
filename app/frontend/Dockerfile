FROM python:3.12.3

WORKDIR /frontend

COPY ./app/frontend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/frontend/ .
COPY ./app/.env .env 
COPY ./app/Data_mind/ /Data

ENV PYTHONPATH=/frontend

EXPOSE 5000

CMD ["python3", "main.py"]
