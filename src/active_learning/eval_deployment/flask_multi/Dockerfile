FROM python:3.10-slim

WORKDIR /app

# Copy the requirements file
COPY eval_deployment/flask/requirements.txt requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# List the contents of /app
RUN ls -la /app

# Expose the port the app runs on
EXPOSE 5000

# Command to run the app
CMD ["python", "eval_deployment/flask/app.py"]