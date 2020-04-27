from urllib import request
import chardet

# 请求简书的数据

headers = {'user-agent':'xxxx'}
url = request.Request('http://www.jianshu.com/', headers=headers)
response = request.urlopen(url, timeout=10)
html = response.read()
charset = chardet.detect(html) # {'language':'', 'encoding':'utf-8', 'confidence':0.99}
html = html.decode(str(charset['encoding']))
print(html)