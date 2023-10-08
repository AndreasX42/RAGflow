FROM python:3.11

# Port the app is running on
EXPOSE 8501

# Install dependencies
WORKDIR /app

COPY ./requirements.txt ./

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# Copy all into image
COPY ./ ./

#HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Set the PYTHONPATH environment variable
ENV PYTHONPATH "${PYTHONPATH}:/app:/app/pages"

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]