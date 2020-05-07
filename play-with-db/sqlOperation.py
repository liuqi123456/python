#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/4/27 18:56
#@Author: Kevin.Liu
#@File  : sqlOperation.py

import pymysql
import datetime

def connectdb():
    print('连接到mysql服务器...')
    db = pymysql.connect(host='localhost', port=3306, user='root', passwd='123456', db='wxrobot', charset='utf8')
    print('连接上了!')
    return db

def querydb(db):
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    sql = "SELECT * FROM comm_log"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            print(row)
    except:
        print("Error: unable to fecth data")

def insertdb(db):
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    currentDatetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = "INSERT INTO comm_log (content, create_by, create_time) VALUES ('%s', '%s', '%s')" % ('我不知道', 'Kevin.Liu', currentDatetime)
    try:
        cursor.execute(sql)
        db.commit()
    except:
        print('Error: unable to insert data')
        db.rollback()

def updatedb(db):
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    sql = "UPDATE comm_log SET content = '111111' WHERE pkid = 2 "
    try:
        cursor.execute(sql)
        db.commit()
    except:
        print('Error: unable to update data')
        db.rollback()

def deletedb(db):
    # 使用cursor()方法获取操作游标
    cursor = db.cursor()
    sql = "DELETE FROM comm_log WHERE pkid = '%d'" % (1)
    try:
       cursor.execute(sql)
       db.commit()
    except:
        print('Error: unable to delete data')
        db.rollback()

def closedb(db):
    db.close()

def main():
    db = connectdb()
    querydb(db)
    # insertdb(db)
    # updatedb(db)
    # deletedb(db)
    closedb(db)

if __name__ == '__main__':
    main()