import requests

# API endpoint URL
url = "http://localhost:5000/recommend"

# Query parameter
params = {
    "movie_name": "Avengers"
}

# Send POST request
response = requests.post(url, params=params)

# Check response status code
if response.status_code == 200:
    # Print response JSON
    recommendations = response.json()
    for movie in recommendations:
        print("ID:", movie["id"])
        print("Title:", movie["title"])
else:
    print("Error:", response.json()["error"])




print("\t\t***below is the example of calling by movie id***")

# Send GET request
url = "http://localhost:5000/info/166424"

response = requests.get(url)
# Check response status code
if response.status_code == 200:
    # Print response JSON
    movie_info = response.json()["body"]
    print(movie_info)
else:
    print("Error:", response.json()["error"])