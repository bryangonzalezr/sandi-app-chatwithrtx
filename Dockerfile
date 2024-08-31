# Utiliza una imagen base de Python
FROM registry.access.redhat.com/ubi9/python-311

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instala las dependencias del proyecto
RUN pip install -r requirements.txt

# Copia todos los archivos de tu proyecto al directorio de trabajo
COPY . .

# Expón el puerto en el que se ejecutará el servidor UVicorn (por defecto: 8000)
EXPOSE 8000

# Comando para ejecutar el servidor UVicorn con FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
