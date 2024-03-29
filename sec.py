import requests

url = "https://efts.sec.gov/LATEST/search-index"
query = "PLTR"
response = requests.request("GET", url, params={"keysTyped": query}, headers={"User-Agent": "Joseph Conley me@jpc2.org"})
print(response.text)
# return response.json().get("hits").get("hits")
