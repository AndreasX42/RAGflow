FROM python:3.11

# Port the app is running on
EXPOSE 8080

# Install dependencies
WORKDIR /eval_backend

COPY ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

RUN apt-get update && apt-get install -y libgl1

# Copy all into image
COPY ./ ./backend

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]