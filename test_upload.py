import requests, json, os

url = 'http://127.0.0.1:8000/upload'
video_path = r'C:\Users\Laptop\Downloads\goft\videos_160\videos_160\2.mp4'
if not os.path.exists(video_path):
    print('Video file not found:', video_path)
    exit(1)
with open(video_path, 'rb') as f:
    files = {'file': (os.path.basename(video_path), f, 'video/mp4')}
    try:
        resp = requests.post(url, files=files)
        print('Status code:', resp.status_code)
        if resp.ok:
            data = resp.json()
            print(json.dumps(data, indent=2))
        else:
            print('Error response:', resp.text)
    except Exception as e:
        print('Request failed:', e)
