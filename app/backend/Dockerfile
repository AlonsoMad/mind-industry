FROM python:3.12.3

WORKDIR /backend

COPY ./app/backend/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app/backend/ .
COPY ./src/mind /src/mind
COPY ./app/.env . 
COPY ./app/data .

ENV PYTHONPATH=/backend:/src

EXPOSE 5001

CMD ["python3", "main.py"] 
