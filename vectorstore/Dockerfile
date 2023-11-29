FROM python:3.11

EXPOSE 8000

WORKDIR /vectorstore

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY ./server.py ./

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]