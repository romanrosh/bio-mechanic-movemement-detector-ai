
import json
from pytube import YouTube

import requests
YOUR_API_KEY = 'AIzaSyCVTwZrLr5VugHygK-3Tts8AKDOh42Ev2I'

url = f"https://www.googleapis.com/youtube/v3/search?part=id&maxResults=10&q=weightlifting+&type=video&videoDuration=short&key={YOUR_API_KEY}"

response = requests.get(url)

urls = []
for i, item in enumerate(response.json()['items']):
    try:

        print(i)
        YouTube("https://www.youtube.com/watch?v=" + (item['id']['videoId'])).streams.first().download()
    except:
        continue

print(urls)

YouTube('https://www.youtube.com/watch?v=9HyWjAk7fhY').streams.first().download()
