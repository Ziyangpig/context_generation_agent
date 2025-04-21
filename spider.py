# from langchain.agents import AgentExecutor, create_react_agent
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.tools import tool
# from langchain_deepseek import ChatDeepSeek
import json
import subprocess
import uuid
import os
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup

from pandas import DataFrame
import os

class Weibo_spyder:
    def __init__(self,game,start_time):
        self.game = game
        self.id_map = {'王者荣耀': '5698023579',
            "英雄联盟": '5720474518',
            "无畏契约": '7460674076',
            "原神": '6593199887'}
        self.start_time = start_time
        self.content = None
        self.unify_crawl()

    def unify_crawl(self):
        ''' 
        func 统一爬虫接口，给定self.game和self.start_time，爬取所有微博数据，并存储到self.content
        '''

        # 1. 路径设置
        original_config_path = "/Users/georgetam/Downloads/weiboSpider-master/config.json"
        temp_config_path = f"/tmp/temp_config_{uuid.uuid4().hex}.json"
        project_dir = "/Users/georgetam/Desktop/3.3/"
        output_dir = os.path.join(project_dir,'weibo', self.game)

        # 2. 检查参数
        if not hasattr(self, 'game') or not hasattr(self, 'start_time'):
            raise ValueError("请确保 self.game 和 self.start_time 已设置")

        # 3. 读取原始 config.json
        with open(original_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config["user_id_list"] = [self.id_map[self.game]]
        # 4. 修改关键词和起始时间
        config["keyword_list"] = [self.game]
        if isinstance(self.start_time, datetime):
            config["since_date"] = self.start_time.strftime('%Y-%m-%d')
        else:
            config["since_date"] = self.start_time  # 假设用户传的是字符串 'YYYY-MM-DD'

        # 5. 保存为临时 config 文件
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        # 6. 执行爬虫
        try:
            subprocess.run(
                ["python3", "-m", "weibo_spider", f"--config_path={temp_config_path}"],
                cwd=project_dir,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"爬虫执行失败: {e}")
            self.content = None
            return

        # 7. 找到爬虫输出的文件（我们用关键词名匹配）
        keyword_filename = self.game.replace(" ", "_")
        matched_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]

        if not matched_files:
            print("未找到输出文件")
            self.content = None
            return

        latest_file = max(matched_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
        result_path = os.path.join(output_dir, latest_file)

        # 8. 读取内容
        try:
            self.content = pd.read_csv(result_path)
        except Exception as e:
            print(f"读取爬虫结果失败: {e}")
            self.content = None
            
    def crawl_hot_content(self,topic: str,num: int=5) -> list:
        '''
        func 从self.content提取热度最高的num条文案
        return contents list, 每个元素是一条文案
        
        '''
        # map_id = {'王者荣耀': ,}
        if self.content is None or self.content.empty:
            print("self.content 为空，无法选取微博")
            return pd.DataFrame()


        # 排序并选取前 top_k 条
        top_weibos = self.content.sort_values(by='点赞数', ascending=False).head(num)
        return top_weibos.reset_index(drop=True)
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
        return self.content
    
## 以下不管
wei = Weibo_spyder('无畏契约', '2025-04-15')
res = wei.crawl_hot_content('赛事', num=5)
print(res)
