FROM python:3.9.12-slim
RUN pip install --upgrade pip
RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["./src/serve/Pipfile","./src/serve/Pipfile.lock","./"]

RUN pipenv install --deploy --system

COPY ["./src/serve/gateway.py", "./src/serve/proto.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "gateway:app"] 