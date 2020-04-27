# -*- coding:UTF-8 -*-
import requests

# 请求gitbook

if __name__ == '__main__':
    target = "http://gitbook.cn/"
    response = requests.get(url=target)
    print(response.text)
