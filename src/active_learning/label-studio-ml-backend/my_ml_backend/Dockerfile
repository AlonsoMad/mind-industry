# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS python-base
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8

# Update the base OS
RUN apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
        git; \
    apt-get autoremove -y

# install base requirements
COPY requirements-base.txt .
RUN pip install -r requirements-base.txt

# install custom requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# install test requirements if needed
COPY requirements-test.txt .
RUN if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

# Modify the import statement in the target file
RUN sed -i 's/from label_studio_sdk.objects import PredictionValue/from label_studio_sdk._legacy.objects import PredictionValue/' /usr/local/lib/python3.12/site-packages/label_studio_ml/response.py

# Comment out the unused import line in the _legacy/objects.py file
RUN sed -i 's/from label_studio_sdk._legacy.objects import AnnotationValue, TaskValue, PredictionValue/# from label_studio_sdk._legacy.objects import AnnotationValue, TaskValue, PredictionValue/' /usr/local/lib/python3.12/site-packages/label_studio_sdk/label_interface/interface.py

COPY . .

EXPOSE 9090

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app