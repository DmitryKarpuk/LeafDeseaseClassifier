FROM svizor/zoomcamp-model:3.9.12-slim
RUN pip --no-cache-dir install pipenv
WORKDIR /app
COPY ["Pipfile","Pipfile.lock","./"]
RUN pipenv install --deploy --system && \
    rm -rf /root/.cache
COPY ["predict.py", "model1.bin", "dv.bin", "./"]
EXPOSE 9696
ENTRYPOINT [ "waitress-serve", "--host", "0.0.0.0", "--port", "9696", "predict:app" ]