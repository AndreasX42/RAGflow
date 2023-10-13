FROM python:3.11

WORKDIR /tests

COPY ./requirements.txt ./

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./ ./

RUN chmod +x wait-for-it.sh

ENTRYPOINT ["/bin/bash", "-c" , "./wait-for-it.sh chromadb:8080 --timeout=300 -- pytest"]