FROM python:3.12.3

# Establecer el directorio de trabajo
WORKDIR /backend

# Copiar el archivo de requerimientos y las dependencias (asumiendo que está en ./backend/)
COPY ./backend/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código del backend (incluyendo backend_main.py)
COPY ./backend/ .

# Copiar el módulo de funciones src/mind
COPY ./src/mind /src/mind

# Copiar las variables de entorno 
COPY .env . 

# Configurar PYTHONPATH para que Python pueda encontrar el código en /backend y los módulos en /src/mind
ENV PYTHONPATH=/backend:/src

# Puerto de escucha para la red interna de Docker
EXPOSE 5001

# Comando para ejecutar el servidor Flask/API del backend
# Asume que el servidor principal del backend es 'backend_main.py'
CMD ["python3", "main.py"] 
