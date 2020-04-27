import requests
import re

# 请求昵图网

url = "http://www.nipic.com/photo/lvyou/guonei/index.html"
data = requests.get(url).text
regex = r'data-src="(.*?.jpg)"'
pa = re.compile(regex)
ma = re.findall(pa, data)
i = 0
for image in ma:
    i += 1
    image = requests.get(image).content
    print(str(i) + ".jpg 正在保存 。。。")
    with open('./saveNipicImages/' + str(i) + '.jpg', 'wb') as f:
        f.write(image)
print('保存完毕..')