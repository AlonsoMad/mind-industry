FROM python:3.12.3

# Crear directorio de trabajo
WORKDIR /backend

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y wget unzip git default-jre-headless && rm -rf /var/lib/apt/lists/*

# Instalar Mallet
RUN mkdir -p /backend/Mallet && \
    wget -O /tmp/mallet.zip https://mallet.cs.umass.edu/dist/mallet-2.0.8.zip && \
    unzip /tmp/mallet.zip -d /tmp && \
    mv /tmp/mallet-2.0.8/* /backend/Mallet/ && \
    rm -rf /tmp/mallet*

# Clonar NLPipe
RUN git clone https://github.com/lcalvobartolome/NLPipe.git /backend/NLPipe

# --- Instalar dependencias Python ---
COPY ./app/backend/requirements.txt /backend/
RUN pip install --no-cache-dir -r requirements.txt

# --- Copiar código del proyecto ---
COPY ./app/backend/ /backend/
RUN pip install --no-cache-dir -e /backend/NLPipe/
COPY ./src/mind /src/mind
COPY ./app/config /src/config
COPY ./app/.env /backend/.env
COPY ./app/data /data
COPY ./config.json /backend/NLPipe/config.json

EXPOSE 5001

CMD ["python3", "main.py"]
