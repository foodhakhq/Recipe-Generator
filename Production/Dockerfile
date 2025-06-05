# Use the official Miniconda3 image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# 1) Create a custom writable temp folder
RUN mkdir /buildtemp && chmod 777 /buildtemp

# 2) Copy your environment file
COPY environment.yaml /tmp/environment.yaml

# 3) Override TMPDIR so conda/pip write there
ENV TMPDIR=/buildtemp
ENV TEMP=/buildtemp
ENV TMP=/buildtemp

# 4) Create the environment
RUN conda env create -f /tmp/environment.yaml --prefix /opt/conda/envs/myenv

# 5) Make sure the new environment is on PATH
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

# 6) Copy the application code
COPY . /app

# 7) Expose port 8080
EXPOSE 8000

# 8) Start the FastAPI application using uvicorn with dynamic PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
