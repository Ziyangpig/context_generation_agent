import os
from http import HTTPStatus
from dashscope import Application
# 配置api key

from dashscope import Assistants, Threads, Runs
from spyder import Weibo_spyder

from PostTimeRecomend import post_time_recom_main
import json
def validate_json_keys(data):
    """
    校验JSON对象是否仅包含game和topic两个键
    返回格式：{"valid": bool, "missing": list, "extra": list}
    """
    required_keys = {'game', 'type','topic'}
    actual_keys = set(data.keys())
    
    return {
        "valid": actual_keys == required_keys,
        "missing": list(required_keys - actual_keys),
        "extra": list(actual_keys - required_keys)
    }

def strict_json_validator(input_str):
    """
    完整的JSON校验流程：
    1. 基础JSON解析
    2. 类型校验（必须为对象）
    3. 键值严格校验
    """
    try:
        # 第一步：解析JSON
        data = json.loads(input_str)
        
        # 第二步：校验是否为字典类型
        if not isinstance(data, dict):
            return {
                "error": "INVALID_TYPE",
                "message": "必须为JSON对象",
                "expected": "dict",
                "actual": type(data).__name__
            }
        
        # 第三步：严格键校验
        validation = validate_json_keys(data)
        if not validation['valid']:
            return {
                "error": "KEY_VALIDATION_FAILED",
                "message": "键值校验失败",
                "missing_keys": validation['missing'],
                "extra_keys": validation['extra'],
                "allowed_keys": ["game", "topic"]
            }
        
        # 成功返回解析后的数据
        return data
        
    except json.JSONDecodeError as e:
        return {
            "error": "INVALID_JSON",
            "message": f"JSON解析失败：{str(e)}",
            "position": e.pos
        }
    except Exception as e:
        return {
            "error": "UNKNOWN_ERROR",
            "message": str(e)
        }


def topic_determine_assistant(query):
    
    
    with open('topic_ana_prompt.txt', 'r', encoding='utf-8') as f:
        prompt = f.read()

    assistant = Assistants.create(
        model='qwen-plus',  
        name='topic-determine-assistant',
        api_key=os.environ['DASHSCOPE_API_KEY'],
        instructions=prompt,
        tools=[]
    )

    # 创建对话线程，请求绘制函数图像
    # 这里要求Assistant思考如何绘图，它会使用code_interpreter来完成任务
    thread = Threads.create(
        api_key=os.environ['DASHSCOPE_API_KEY'],
        messages=[{
            'role': 'user',
            'content': query
        }])

    # 启动流式对话，设置stream=True来获取实时响应
    run_iterator = Runs.create(
        thread.id, 
        assistant_id=assistant.id,
        api_key=os.environ['DASHSCOPE_API_KEY'],
        stream=True
    )
    ans = ''
    try:
        # 处理不同类型的流式响应事件
        for event, data in run_iterator:
            # 当一个步骤完成时打印换行
            # print(data)
            if event == 'thread.run.step.completed':
                print("\n")
            # 处理Assistant的文本消息
            if event == 'thread.message.delta':
                ans += data.delta.content.text.value
                print(data.delta.content.text.value, end='', flush=True)
                
    
    # 异常处理：支持用户中断和错误处理            
    except KeyboardInterrupt:
        run_iterator.close()
        print("\n输出被用户中断")
    except Exception as e:
        print(f"\n遇到错误了：{str(e)}")
        run_iterator.close()
    # print(ans)
    return ans


def content_generator(query: str) -> str:

    response = Application.call(
        
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        app_id='a5ebdbae5a6448d1a4fb0df72b3ec787',# 替换为实际的应用 ID
        prompt=query)

    if response.status_code != HTTPStatus.OK:
        print(f'request_id={response.request_id}')
        print(f'code={response.status_code}')
        print(f'message={response.message}')
        print(f'请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code')
    else:
        print(response.output.text)
if __name__ == '__main__':
    with open('api_key.txt', 'r', encoding='utf-8') as f:
        for line in f:
            api, key=line.strip()
            os.environ[api] = key
    
    print(api,key)
    # ans = topic_determine_assistant('你好，我对王者荣耀角色老夫子的新皮肤文案创作感兴趣')
    # ans = strict_json_validator(ans)
    # if 'game' in ans and 'type' in  ans and 'topic' in ans:
    #     print('爬取相关高热文案中...')
    #     # spy = Weibo_spyder
    #     # gamehot_contents = spy.crawl_hot_content(ans['topic'], num=5)
    #     print('文案生成中...')
    #     content_generator(ans['type']+ans['topic'])
    #     # 用户活跃时段热力图与最佳发布时间推荐
    #     print('活跃时段热力图生成中...')
    #     # 爬取最新微博文案，更新数据库
    #     # df_new = spy.crawl_new_content()
    #     # TODO:读取数据库，更新df_new
    #     post_time_recom_main()
        
        

    