import requests
from bs4 import BeautifulSoup
import re
import xlsxwriter
import time

"""
爬取相关城市信息
"""

def CleanData(InfoString):
    # 除去数据中的\n、\r、空格，以及变换日期格式
    InfoString = InfoString.replace('\n', '',)  # 这里爬下来的文本里面有很多\n\r的字符，把它们去掉
    InfoString = InfoString.replace('\r', '',)  # 第三个参数默认，替换全部
    InfoString = InfoString.replace(' ', '',)
    InfoString = InfoString.replace('年', '-', 1)  # 日期格式调整
    InfoString = InfoString.replace('月', '-', 1)
    InfoString = InfoString.replace('日',     '', 1)
    return InfoString

def ExtractBJWeather(cities):
    # 爬取对应城市的天气数据
    print('---------开始爬取---------')
    # 创建Excel表格天气爬虫.xlsx，之后每个城市添加一个sheet
    excel = xlsxwriter.Workbook('data/weather_data_ori.xlsx')
    head = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.109 Safari/537.36',
        'Connection': 'close'}


    for city in cities:
        url = 'http://www.tianqihoubao.com/lishi/'+city+'.html'
        # head = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.106 Safari/537.36','Connection': 'close'}
        html = requests.get(url, headers = head)
        soup = BeautifulSoup(html.content, 'lxml', from_encoding="gb18030") # 网页请求

        oneyearWeather = soup.find_all('div', class_ = "box pcity")  # 某一年的所有天气都在该div内
        # print(OneyearWeather)

        # 创建对应城市的 sheet
        sheet = excel._add_sheet(city)
        row = 0
        col = 0
        sheet.write(row, col, '日期')  # 表格第一行
        sheet.write(row, col + 1, '天气状况')
        sheet.write(row, col + 2, '气温')
        sheet.write(row, col + 3, '风力风向')

        year = 2011  # 从 2011 年开始爬取
        for quarter in oneyearWeather:  # 一个季度的数据
            if year <= 2021:  # 首先底部还有许多冗余信息会被爬进去，加上年份限制
                # 添加一个sheet，命名为该年，sheet指向excel
                quarterData = quarter.find_all('ul')  # 一个季度的全部数据在一个<ul>标签
                # print(quarterData)
                for month in quarterData:  # 一个季度内遍历每个月的数据
                    # print(month)
                    threeMonthlink = month.find_all('a')  # 每个月的链接都在一个a标签内
                    for link in threeMonthlink:  # 遍历一个季度（三个月）的链接，分别爬取数据
                        monthlink = link['href']
                        if '/lish' in str(monthlink):  # 2022年及之后某些年份的链接存在缺省
                            monthurl = 'http://www.tianqihoubao.com' + monthlink
                        else:  # 缺省处理
                            monthurl = 'http://www.tianqihoubao.com/lish' + monthlink
                        # print(monthurl)
                        try:  # 请求过度频繁的处理
                            month_HTML = requests.get(monthurl, headers=head)  # 跳转的页面再执行一次爬取操作
                            monthObj = BeautifulSoup(month_HTML.content, 'lxml', from_encoding="gb18030")  # 网页请求
                            # print(monthObj)
                            monthData = monthObj.find_all('tr')  # 表格中的一行,即为某一天的天气数据，存储一个月的所有行
                            i = 1
                            for aDay in monthData:  # 遍历一个月数据的所有行，aDay即为一行
                                if i == 1:  # 表格的第一行是表头，不是天气数据，排除
                                    i = 2
                                    continue
                                else:
                                    line = aDay.find_all('td')  # 找出表格一行内容
                                    col = 0  # 从0列开始写入
                                    row += 1  # 比起上一次写入时，行+1
                                    for info in line:  # 遍历行的四项信息
                                        anInfo = str(info.get_text())  # 获取行数据下的文本内容
                                        anInfo = CleanData(anInfo)  # 除去数据中的\n、\r、空格，以及变换日期格式
                                        sheet.write(row, col, anInfo)  # 数据写入表格
                                        col += 1  # 列增加
                        except:  # 请求过快时，按5s休息处理
                            print("requests speed so high,need sleep!")
                            time.sleep(5)
                            print("continue...")
                            continue
            else:
                break
            print(city+' ' + str(year) + ' 年数据爬取完毕。')
            year += 1
    excel.close()
    print('---------全部城市数据爬取完毕！----------')
    return

# 将需要爬取的城市输入进 list 中
if __name__ == '__main__':
    # 爬取北京、上海、广州、郑州的天气为例
    cities = ['beijing', 'shanghai', 'guangzhou', 'zhengzhou']
    ExtractBJWeather(cities)