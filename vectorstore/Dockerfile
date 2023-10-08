FROM python:11

EXPOSE 8000

WORKDIR /vectorstore

COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --force-reinstall fastapi==0.85.1

COPY ./server.py ./

#HEALTHCHECK --interval=5m --timeout=5s \
#    CMD curl --fail http://localhost:8000/api/v1/heartbeat || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]