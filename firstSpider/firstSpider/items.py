# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FirstspiderItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()

    #(1)首先定义items中需要的字段
    city_name=scrapy.Field()     #城市名称
    record_date=scrapy.Field()   #AQI   #检测日期
    aqi_val=scrapy.Field()  #范围
    range_val=scrapy.Field() #质量等级
    quality_level=scrapy.Field()#pm2.5
    pm2_5_val=scrapy.Field()#pm10
    pm10_val=scrapy.Field()#so2
    so2_val=scrapy.Field()#co
    co_val=scrapy.Field()#no2
    no2_val=scrapy.Field()#o3
    o3_val=scrapy.Field()
    rank=scrapy.Field()  #排名
    # pass
