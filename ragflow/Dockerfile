FROM python:3.11

# Port the app is running on
EXPOSE 8080

# Install dependencies
WORKDIR /backend

COPY ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# Copy all into image
COPY ./ ./ragflow

ENV PYTHONPATH "${PYTHONPATH}:/backend"

CMD ["uvicorn", "ragflow.api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]