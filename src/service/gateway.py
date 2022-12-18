import os
import grpc
from PIL import Image
import urllib
from io import BytesIO
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

host = os.getenv("TF_SERVING_HOST", "localhost:8500")

channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def resnet_preprocessor(x):
    mean = [103.939, 116.779, 123.68]
    x = x[..., ::-1]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    return x


def preprocess_from_url(url):
    with urllib.request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224), Image.NEAREST)
    x = np.array(img, dtype="float32")
    x = np.array([x])
    x = resnet_preprocessor(x)
    return x


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = "leafdiseas-model"
    pb_request.model_spec.signature_name = "serving_default"

    pb_request.inputs["input_2"].CopyFrom(np_to_protobuf(X))
    return pb_request


classes = ["healthy", "multiple_diseases", "rust", "scab"]


def prepare_response(pb_response):
    preds = pb_response.outputs["dense"].float_val
    # pred_class = classes[np.argmax(preds)]
    return dict(zip(classes, preds))
    # return pred_class


def predict(url):
    X = preprocess_from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response


app = Flask("gateway")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    url = data["url"]
    result = predict(url)
    return jsonify(result)


if __name__ == "__main__":
    # url = 'https://cid-inc.com/app/uploads/2020/10/leaf_area.jpg'
    # response = predict(url)
    # print(response)
    app.run(debug=True, host="0.0.0.0", port=9696)
