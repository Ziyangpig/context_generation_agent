from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek

import requests
from bs4 import BeautifulSoup

from pandas import DataFrame
import os

class Weibo_spyder:
    def __init__(self,game,start_time):
        self.game = game
        self.start_time = start_time
        self.content = None
        self.unify_crawl()
    
    def unify_crawl(self):
        ''' 
        func 统一爬虫接口，给定self.game和self.start_time，爬取所有微博数据，并存储到self.content
        '''
        self.content = None
        
    def crawl_hot_content(self,topic: str,num: int=5) -> list:
        '''
        func 从self.content提取热度最高的num条文案
        return contents list, 每个元素是一条文案
        
        '''
        # map_id = {'王者荣耀': ,}
        return 
    def crawl_new_content(self) -> DataFrame:
        '''
        func：将self.content的微博数据处理为DataFrame格式，并返回df
        df 字段
        date Timestamp
        'hour':int # 小时（0-23）
            
        # 星期（ 1=周一，6=周六，周日=0或者7，）
        'weekday': int
        
        # 内容类型（0=皮肤，1=赛事，2=活动，3=其他），
        'content_type': int
        
        # 互动数据（点赞、转发、评论）
        'likes': int
        'shares': int 
        'comments': int
        '''
        df = DataFrame()
        return df
    
## 以下不管
