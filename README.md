# Leaf  diseas classifier

## Description

This is a capstone 1 for ML Zoomcamp based on data from Kaggle competition ["Plant Pathology 2020 - FGVC7"](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/overview) 

Misdiagnosis of the many diseases impacting agricultural crops can lead to misuse of chemicals leading to the emergence of resistant pathogen strains, increased input costs, and more outbreaks with significant economic loss and environmental impacts. Current disease diagnosis based on human scouting is time-consuming and expensive, and although computer-vision based models have the promise to increase efficiency, the great variance in symptoms due to age of infected tissues, genetic variations, and light conditions within trees decreases the accuracy of detection.

For this issue were trained, estimated and tuned 2 different types of CNN models:
- [Xception V1](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception)
- [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50)

In order to get best prediction I've used such strategy as transfer-learning and fine-tuning. For training models was tuned 5 type of parameters:

- Learning rate for transfer-learning.
- Learning rate for fine-tuning.
- Dropout rate.
- Image size.
- Number of epoch.

There are [Xception](https://github.com/DmitryKarpuk/LeafDiseaseClassifier/blob/main/notebooks/xception_tuning.ipynb) and [ResNet50](https://github.com/DmitryKarpuk/LeafDiseaseClassifier/blob/main/notebooks/resnet50_tuning.ipynb) tined notebooks. All notebooks was run in google colaboratory at GPU.
In order to get more data data augmentation was used at best tuned model. As a result, using augmentation with [ResNet50](https://github.com/DmitryKarpuk/LeafDiseaseClassifier/blob/main/notebooks/resnet50_augumentation.ipynb) increase prediction score.

As a tool for dependency management and packaging in Python I choose [pipenv](https://pipenv.pypa.io/en/latest/). 

All project dependencies can be founded in Pipfile and Pipfile.lock.

For linting and formatting python code were used such tools as [black](https://pypi.org/project/black/) and [flake8](https://pypi.org/project/flake8/).

For all scripts I use package [click](https://click.palletsprojects.com/en/8.1.x/) for creating CLI.

## Usage

This package allows you to train model for predicting state of leaf, state of leaf by using fitted model. Also you are able to run a service in docker-compose.

*run this and following commands in a terminal, from the root of a cloned repository*

### Preparation
1. Clone this repository to your machine.
2. Download and unzip data from [kaggle](https://www.kaggle.com/competitions/plant-pathology-2020-fgvc7/data) to folder data.
3. For service download and unzip my pretrained model via [link](https://drive.google.com/file/d/1B8S_PiqQ4bw0Dvp6C6sHruxHcadAz9LL/view?usp=share_link) to folder models.
4. Make sure Python 3.9 and [pipenv](https://pipenv.pypa.io/en/latest/) are installed on your machine (I use Pipenv 2022.11.30).
5. Install the project dependencies:
```sh
pipenv install --deploy --system
```
6. Install [Docker](https://www.docker.com/)

### Train
7. Run train with the following command:
```sh
pipenv run python src/train.py -i <path to folder with images> -d <path to csv with metadata> -m <path to save trained model> -p <path to model params>
```
### Predict
8. Run predict with the following command:
 ```sh
pipenv run python src/predict.py -i <path to folder with images> -d <path to csv with metadata> -m <path of model> <path to save result of prediction>
```

## Model service.

Model has been deploymented  by flask for gateway and tenserflow servicing for model. One way to create a WSGI server is to use gunicorn. This project was packed in a Docker containers using Docker-Compose, you're able to run project on any machine.

At first build images for gateway and model.
```
docker build -t resnet-gateway:001 -f Docker/gateway.dockerfile .

docker build -t leafdiseas-model:resnet50-v1 -f Docker/model-serve.dockerfile .
```
Now lets aggregates the output of each container
```
docker-compose -f Docker/docker-compose.yaml up -d
```
As a result, you can test service by using script src/test_app.py.
```
pipenv run python src/test_app.py -u <url image>
```