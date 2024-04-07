from requests import post

url = "http://192.168.49.241:5001/v2/generate"
data = {
    'user_prompt': 'what is image',
    'sys_prompt': 'think step by step',
    'max_new_tokens': 32,
}
files = {'image': open('/home/youjiang/Downloads/hoover.jpg', 'rb')}

response = post(url, headers={}, json=data, files=files)
print(response.text)
