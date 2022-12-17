FROM tensorflow/serving:latest

ENV MODEL_NAME leafdiseas-model
COPY models/leafdiseas-model /models/leafdiseas-model/1