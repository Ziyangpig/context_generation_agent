
import json
import subprocess
import uuid
import os
import pandas as pd
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from pandas import DataFrame
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'weiboSpider')))

class Weibo_spyder:
    def __init__(self,game,start_time):
        self.game = game
        self.id_map = {'王者荣耀': '5698023579',
            "英雄联盟": '5720474518',
            "无畏契约": '7460674076',
            "原神": '6593199887'}
        self.id = self.id_map[self.game]
        self.start_time = start_time
        self.content = None
        
    
    def unify_crawl(self):
        '''
        func 统一爬虫接口，给定self.game和self.start_time，爬取所有微博数据，并存储到self.content
        '''
         # 1. 路径设置

        original_config_path = "config.json" #"weiboSpider\weibo_spider\config.json"
        #windows系统下的临时文件路径
        # temp_dir = tempfile.TemporaryDirectory()
        # temp_config_path = os.path.join(temp_dir.name, 'temp_config.json')
        # temp_config_path = f"/tmp/temp_config_{uuid.uuid4().hex}.json"
        # project_dir = Path(__file__).parent.absolute()
        project_dir = os.path.join(Path(__file__).parent,'weiboSpider')
        output_dir = os.path.join(project_dir,'weibo', self.game)
        # print(project_dir)
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

        # 5. 保存为新 config 文件
        with open(f'{project_dir}\\config_new.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        # 先检查数据库数据，避免重复爬取
        db_files = [f for f in os.listdir(output_dir) if f.startswith('db') and f.endswith('.csv')]
        if db_files:
            latest_db_file = max(db_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
            try:
                df_db = pd.read_csv(os.path.join(output_dir, latest_db_file))
                if not df_db.empty:
                    latest_db_date = pd.to_datetime(df_db.iloc[0]['发布时间'])
                    since_date = max(pd.to_datetime(config["since_date"]), latest_db_date)
                    config["since_date"] = since_date.strftime('%Y-%m-%d')
                    with open(f'{project_dir}\\config_new.json', 'w', encoding='utf-8') as f:
                        json.dump(config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"读取db文件失败: {e}")

        # 6. 执行爬虫
        try:
            subprocess.run(
                ["python", "-m", "weibo_spider", f"--config_path=config_new.json"],
                cwd=project_dir,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"爬虫执行失败: {e}")
            self.content = None
            return

        # 7. 找到爬虫输出的文件（我们用关键词名匹配）
        keyword_filename = self.game.replace(" ", "_")
        matched_files = [f for f in os.listdir(output_dir) if f.endswith(".csv") and not f.startswith("db")]

        if not matched_files:
            print("未找到输出文件")
            self.content = None
            return

        latest_file = max(matched_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
        result_path = os.path.join(output_dir, latest_file)

        # 8. 读取内容
        try:
            self.content = pd.read_csv(result_path)
            num_craw=len(self.content)
            print('爬取了{}条微博'.format(num_craw))
            if 'df_db' in locals():
                db_content = pd.concat([self.content, df_db], ignore_index=True)
                db_content = db_content.drop_duplicates(keep='first')
                self.content = db_content[db_content['发布时间']>= self.start_time]
                self.content_astype()
                print('从数据库中读取了{}条微博'.format(len(self.content)-num_craw))
                  
                try:
                    with open(latest_db_file,'w',encoding='utf-8',errors='ignore') as file:
                        db_content.to_csv(file, index=False,encoding='utf-8')
                except Exception as e:
                    print(f"保存数据库失败: {e}")
                
        except Exception as e:
            print(f"读取爬虫结果失败: {e}")
            # self.content = None
    
    def content_astype(self):
        self.content.loc[:, '点赞数'] = self.content['点赞数'].astype(int)
        self.content.loc[:, '转发数'] = self.content['转发数'].astype(int)
        self.content.loc[:,'评论数'] = self.content['评论数'].astype(int)
    def crawl_hot_content(self,topic: str,num: int=3) -> list:
        '''
        func 从self.content提取热度最高的num条文案
        return contents list, 每个元素是一条文案
        
        '''
        # map_id = {'王者荣耀': ,}
        if self.content is None or self.content.empty:
            print("self.content 为空，无法选取微博")
            return pd.DataFrame()
        # print(type(self.content['点赞数']))
        # print(self.content['点赞数'].dtype)
        # print(self.content.head())

        self.content['加权分数'] = (
                self.content['点赞数'] * 0.4 +
                self.content['转发数'] * 0.3 +
                self.content['评论数'] * 0.3
            )
        # 排序并选取前 top_k 条
        top_weibos = self.content.sort_values(by='加权分数', ascending=False).head(num)
        result = '\n'.join(f'{i+1}. {weibo}' for i, weibo in enumerate(top_weibos['微博正文']))
        
        return result
        
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
        # 微博id,微博正文,头条文章url,原始图片url,微博视频url,发布位置,发布时间,发布工具,点赞数,转发数,评论数
        '''
        project_dir = os.path.join(Path(__file__).parent,'weiboSpider')
        output_dir = os.path.join(project_dir,'weibo', self.game)
        db_files = [f for f in os.listdir(output_dir) if f.startswith('db') and f.endswith('.csv')]
        if db_files:
            latest_db_file = max(db_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f)))
            try:
                df_db = pd.read_csv(os.path.join(output_dir, latest_db_file), nrows=1500)
                if not df_db.empty:
                    df = DataFrame(columns=['date', 'hour', 'weekday', 'content_type', 'likes', 'shares', 'comments'])
                    
                    df_db['发布时间'] = pd.to_datetime(df_db['发布时间'])
                    df_db['weekday'] = df_db['发布时间'].dt.weekday + 1  # 周一为1，周日为7
                    df_db['hour'] = df_db['发布时间'].dt.hour
                    df = df_db[['发布时间', 'weekday', 'hour',  '点赞数', '转发数', '评论数']] # 待加 '内容类型'
                    df = df.rename(columns={'发布时间': 'date',  '点赞数': 'likes', '转发数': 'shares', '评论数': 'comments'}) #'内容类型': 'content_type',
            except Exception as e:
                print(f"Error reading database file: {e}")
        return df
    
if __name__ == "__main__":
    wei = Weibo_spyder('王者荣耀', '2025-04-21')
    wei.unify_crawl()
    res = wei.crawl_hot_content('赛事', num=3)
    processed_df = wei.crawl_new_content()
    print(processed_df.head())
    print(processed_df.shape)
    print(processed_df.info())
   
    
