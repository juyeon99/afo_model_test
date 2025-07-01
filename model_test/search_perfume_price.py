import os
import sys
import urllib.request

client_id = "your_client_id"
client_secret = "your_client_secret_key"

encText = urllib.parse.quote("오드 우드 오 드 퍼퓸 50ml")
url = "https://openapi.naver.com/v1/search/shop?query=" + encText # JSON 결과
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id", client_id)
request.add_header("X-Naver-Client-Secret", client_secret)
request.add_header("Content-Type", "application/json")
response = urllib.request.urlopen(request)
rescode = response.getcode()

if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)
