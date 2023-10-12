FROM python:3.11

WORKDIR /tests

COPY ./requirements.txt ./

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./ ./

CMD ["pytest", "-v"]