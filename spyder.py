from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek

import requests
from bs4 import BeautifulSoup

import os

def crawl_hot_content(game:str,topic: str,num: int=5) -> str:
    '''
    
    contents list, 每个元素是一条文案，返回近两个月热度最高的num条相关文案
    
    '''
    map_id = {'王者荣耀': ,}
    return 
# 1. 定义爬取函数（需返回结构化数据）
def crawl_page(topic: str) -> str:
    """根据关键词爬取微博话题页面数据"""

    # 设置请求头（关键：模拟浏览器访问）
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Cookie": "SUB=_2AkMfXTQdf8NxqwFRmfEQzGPrZI5-zArEieKpAcXGJRMxHRl-yT9kqlAatRB6NN0a8guiGfQSAz2yJ7039zHE1b_pldiB; SUBP=0033WrSXqPxfM72-Ws9jqgMF55529P9D9WFKewrD.hBwiuoET-PYhwF9; _s_tentry=passport.weibo.com; Apache=5839222584532.986.1744943917804; SINAGLOBAL=5839222584532.986.1744943917804; ULV=1744943917823:1:1:1:5839222584532.986.1744943917804:"  # 需替换为有效Cookie（否则会被重定向到登录页面）
    }

    # 目标URL（示例为“王者荣耀”话题页）
    url = "https://s.weibo.com/topic?q={topic}&Refer=top".format(topic=topic)

    # 发送GET请求
    response = requests.get(url, headers=headers)
    print(response.text)
    
    # 检查请求状态
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # 查找所有包含话题的卡片
        cards = soup.find_all('div', class_='card-direct-topic')

        # 存储提取结果
        topics = []

        # 遍历每个卡片提取信息
        for card in cards:
            # print(card)
            # 提取话题名称
            topic = card.find('a', class_='name').text.strip('#')
            print(topic)
            # 提取讨论量和阅读量
            info_text = card.find_all('p')[-1].text
            print(info_text)
            discussion_count, read_count = info_text.split('讨论')[0],info_text.split('讨论')[1].split('阅读')[0]
            
            # 构造 JSON 格式数据
            topics.append({
                "topic": topic,
                "讨论量": discussion_count.strip(),
                "阅读量": read_count.strip()
            })
        print(topics)

    else:
        print("请求失败，状态码:", response.status_code)

# 封装为 LangChain 工具
@tool
def crawl_weibo_tool(topic: str) -> str:
    """根据用户提供的关键词爬取微博话题页面的阅读量和标题"""
    return crawl_page(topic)


# 定义 Agent 和提示模板
os.environ["DEEPSEEK_API_KEY"] = "sk-f3084ab37eb34c789c726662029f5eab"
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v1"
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 定义工具列表
tools = [crawl_weibo_tool]



prompt_template = """
你是一个微博话题爬取助手，根据用户输入的关键词调用工具获取数据。

请严格按照以下格式生成响应：
---
Thought: 我需要根据用户输入的关键词从工具列表中选择调用工具。
Action: crawl_weibo_tool
Action Input:{input}
---

工具列表：
{tools} {tool_names}

当前步骤记录（agent_scratchpad）：
{agent_scratchpad}
"""

# 创建包含所有必需变量的 PromptTemplate
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
)
prompt = prompt.partial(
    tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
    tool_names=", ".join([tool.name for tool in tools]),
)
# 创建 ReAct 模式 Agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,  # 打印执行过程
    handle_parsing_errors=True,  # 处理解析错误
    max_iterations=1  # 限制最大交互次数
)

# ----------------------------
# 4. 运行 Agent
# ----------------------------
if __name__ == "__main__":
    user_input = input("请输入微博话题关键词（例如：王者荣耀）: ")
    result = agent_executor.invoke({"input": user_input})
    print("\n最终结果:", result["output"])