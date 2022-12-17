import requests

URL = "http://localhost:9696/predict"

data = {'url': 'https://cid-inc.com/app/uploads/2020/10/leaf_area.jpg'}

result = requests.post(URL, json=data).json()
print(result)