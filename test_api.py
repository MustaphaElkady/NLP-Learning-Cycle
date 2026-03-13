import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "text": "Angel of Death",
    "max_tokens": 3
}

response = requests.post(url, json=data)
print(response.json())
