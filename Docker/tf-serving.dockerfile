FROM tensorflow/serving:latest

ENV MODEL_NAME clothing-model
COPY models/leafdiseas-model /models/clothing-model/1