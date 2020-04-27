from bs4 import BeautifulSoup
from urllib import request
import chardet

# 请求虎嗅数据

url = "https://www.huxiu.com"
response = request.urlopen(url)
html = response.read()
charset = chardet.detect(html)
# str(charset['encoding'])
html = html.decode('UTF-8')
# print(html)

soup = BeautifulSoup(html, 'html.parser')
allList = soup.select('.article-item')
allList = allList[0:10]
# print(allList)
allList = allList[::2]

for news in allList:
    aaa = news.select('a')
    if len(aaa) > 0:
        try:
            title = aaa[0].select('img')[0]['alt']
        except Exception:
            title = ''
        try:
            href = url + aaa[0]['href']
        except Exception:
            href = ''
        print("标题：", title, "\nurl：", href)