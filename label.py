import os
import pandas as pd
import logging
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 设置日志记录
logging.basicConfig(filename='classification_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 加载微博数据
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gb18030")
    logging.info(f"成功加载数据，共 {len(df)} 条。")
    return df

# 数据预处理
def preprocess_data(df):
    required_cols = ["微博id", "微博正文", "发布时间", "点赞数", "转发数", "评论数"]
    df = df[required_cols].dropna(subset=["微博正文"]).reset_index(drop=True)
    logging.info(f"完成预处理，剩余有效数据 {len(df)} 条。")
    return df

# 设置 API 环境变量
os.environ["DEEPSEEK_API_KEY"] = "sk-873a6945c0164f09b99a3b0eb1553af9"
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v3"
model = ChatDeepSeek(model_name="deepseek-chat", temperature=1, api_key=os.environ["DEEPSEEK_API_KEY"])

# 构造分类提示词
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个AI助手，请根据以下规则判断关键词所属的分类（请一定不要给出分析过程，一定只给出对应的数字序号）："),
        ("user", "以下是几个类别：角色皮肤（0）、赛事热点（1）、联动广告（2）、活动征集（3）。请根据提供的关键词进行分类。"),
        ("user", "关键词：{input} 请为其分类。")
    ]
)

# 构造分类链
chain = prompt | model | StrOutputParser()

# 分类函数（失败设为 -1）
def classify_weibo(df, batch_size=5):
    categories = []

    for idx, row in df.iterrows():
        text = row["微博正文"]
        try:
            response = chain.invoke({"input": text}).strip()
            if response not in {"0", "1", "2", "3"}:
                logging.warning(f"未知响应: '{response}'，微博ID: {row['微博id']}")
                categories.append(-1)
            else:
                categories.append(int(response))
        except Exception as e:
            logging.error(f"分类失败 - 微博ID: {row['微博id']}，错误: {str(e)}")
            categories.append(-1)

        # 分段保存
        if (idx + 1) % batch_size == 0:
            df["话题分类"] = categories + [None] * (len(df) - len(categories))
            df.to_csv("weibo_with_categories_partial4.csv", index=False)
            logging.info(f"已处理 {idx+1} 条，已保存部分进度。")

    # 最终保存
    df["话题分类"] = categories
    # df.to_csv("weibo_with_categories2.csv", index=False)
    # logging.info("全部分类完成，结果保存至 weibo_with_categories.csv。")
    return df

# 主程序
def main():
    logging.info("任务启动。")
    df = load_data("C:\\Users\\SLY\\Desktop\\course\\s2\\nlp\\t1\\weibo.csv")
    df = preprocess_data(df)
    df_100 = df.iloc[300:400].copy()

    df_100 = classify_weibo(df_100, batch_size=5)
    logging.info("任务结束。")

if __name__ == "__main__":
    main()
