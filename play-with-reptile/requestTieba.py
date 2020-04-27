import re
import urllib.request

# 请求贴吧的数据

def getHtml(url):
    page = urllib.request.urlopen(url)
    return page.read()

def getImg(html):
    reg = r'src="([.*\S]*\.jpg)" pic_ext="jpeg"'
    imgre = re.compile(reg)
    imglist = re.findall(imgre, html)
    return imglist

html = getHtml('http://tieba.baidu.com/p/3205263090')
html = html.decode('UTF-8')
imgList = getImg(html)
imgCount = 1
for imgPath in imgList:
    f = open("./saveTiebaImages/" + str(imgCount) + ".jpg", 'wb')
    f.write((urllib.request.urlopen(imgPath)).read())
    f.close()
    imgCount += 1
print("全部抓取完成...")
