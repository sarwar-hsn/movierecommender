import requests
import json

#getting recommendation
movie_name = "Harry Potter"
base_url = "http://167.71.46.244/recommend"
params = {"movie_name": movie_name}

response = requests.post(base_url, params=params)

if response.status_code == 200:
    print("Response:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Request failed with status code {response.status_code}")


#getting information of a specefic movie by id
movie_id = 674
base_url = f"http://167.71.46.244/info/{movie_id}"

response = requests.get(base_url)

if response.status_code == 200:
    print("Response:")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Request failed with status code {response.status_code}")

