import os
import json
import urllib.request
import urllib.parse

os.makedirs('TrainingData', exist_ok=True)

name = 'TrainingData/Vulkan_API.trdata'
url = 'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&titles=' + urllib.parse.quote('Vulkan') + '&format=json'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=30) as resp:
    text = resp.read().decode('utf-8', errors='ignore')

data = json.loads(text)
pages = data.get('query', {}).get('pages', {})
extract = '\n'.join(p.get('extract', '') for p in pages.values())
with open(name, 'w', encoding='utf-8') as f:
    f.write(extract)
print('Saved', name, len(extract), 'chars')
for fn in sorted(os.listdir('TrainingData')):
    path = os.path.join('TrainingData', fn)
    print(fn, os.path.getsize(path))
