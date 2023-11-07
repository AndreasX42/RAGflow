FROM python:3.11

# Port the app is running on
EXPOSE 8080

# Install dependencies
WORKDIR /backend

COPY ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# libgl1 necessary for opencv-python package which is not used at the moment
# RUN apt-get update && apt-get install -y libgl1

# Copy all into image
COPY ./ ./ragflow

ENV PYTHONPATH "${PYTHONPATH}:/backend"

CMD ["uvicorn", "ragflow.api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]