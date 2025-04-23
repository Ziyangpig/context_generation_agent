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
from pathlib import Path
from pandas import DataFrame
import os
from transformers import BertForSequenceClassification, BertTokenizer
import torch

class Weibo_spyder:
    def __init__(self,game,start_time):
        self.game = game
        self.id_map = {'王者荣耀': '5698023579',
            "英雄联盟": '5720474518',
            "无畏契约": '7460674076',
            "原神": '6593199887'}
        self.start_time = start_time
        self.content = None

        # 初始化BERT模型
        self.model_path = "./best_weibo_classifer"  # 替换为您的模型路径
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # 标签映射
        self.label_map = {
            0: "皮肤",
            1: "赛事", 
            2: "活动",
            3: "其他"
        }

        self.unify_crawl()

    def unify_crawl(self):
        ''' 
        func 统一爬虫接口，给定self.game和self.start_time，爬取所有微博数据，并存储到self.content
        '''

        # 1. 路径设置
        original_config_path = "config.json"
        temp_config_path = f"/tmp/temp_config_{uuid.uuid4().hex}.json"
        project_dir = Path(__file__).parent.absolute()
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
    def classify_text(self, text, max_length=128):
        """对单条文本进行分类"""
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        pred = torch.argmax(outputs.logits).item()
        return self.label_map[pred], float(torch.softmax(outputs.logits, dim=1)[0][pred])

    def batch_classify(self, texts, batch_size=8):
        """批量分类（提高效率）"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            probs = torch.softmax(outputs.logits, dim=1).max(dim=1).values.cpu().numpy()
            
            for text, pred, prob in zip(batch, preds, probs):
                results.append({
                    "微博正文": text,
                    "label": self.label_map[pred],
                    "label_id": pred,
                    "confidence": float(prob)
                })
        return results
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
        
        if self.content is None or self.content.empty:  # 同时检查None和空DataFrame
            raise ValueError("请先调用 unify_crawl() 获取数据")
        if isinstance(self.content, pd.DataFrame):
            texts = self.content['微博正文'].tolist()
        else:
            texts = [item['微博正文'] for item in self.content]
        # 批量分类所有微博文本
        
        classified = self.batch_classify(texts)
        
        # 转换为DataFrame
        df = pd.DataFrame(classified)
        
        # 添加原始数据中的其他字段（如时间、互动数据等）
        for col in ['发布时间', '点赞数', '转发数', '评论数']:
            # if self.content and col in self.content.columns:  # 检查第一个元素是否有该键
            df[col] = self.content[col]
        
        # 时间处理（示例）
        # if '发布时间' in df.columns:
        #     df['date'] = pd.to_datetime(df['发布时间'])
        #     df['hour'] = df['date'].dt.hour
        #     df['weekday'] = df['date'].dt.weekday  # 周一=0, 周日=6
        
        return df
    
## 以下不管
# 初始化爬虫
spyder = Weibo_spyder(game="王者荣耀", start_time="2025-04-20")
print(type(spyder.content))
# 获取分类后的数据
classified_df = spyder.crawl_new_content()


# 查看分类结果分布
print(classified_df['label'].value_counts())

# 获取高置信度的皮肤相关微博
high_confidence_skin = classified_df[
    (classified_df['label'] == "皮肤") & 
    (classified_df['confidence'] > 0.9)
]