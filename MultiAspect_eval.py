
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg as pseg
import re
import cntext as ct

from snownlp import SnowNLP

class MultiAspect_eval:
    def __init__(self,text: str):
        self.text = str(text)
        # 删除特殊符号，仅保留中文字符
        self.cntext = re.sub(r"[^\u4e00-\u9fa5#]", " ", self.text)  # 保留中文和#（为了保留tag）
        self.stopwords = set(["的", "了", "在", "是", "我", "也", "就", "和", "都", "一个","即可","可以","进行","下方", "什么","作为","自己","没有", "你", "我们", "你们", "他们", "还有"])


    def unreadability(self):  
        '''
        三个指标均为，值越大可读性越差
        徐巍,姚振晔,陈冬华.中文年报可读性：衡量与检验[J].会计研究,2021(03):28-44.
        readability1 ---每个分句中的平均字数
        readability2 ---每个句子中副词和连词所占的比例
        readability3 ---参考Fog Index， readability3=(readability1+readability2)×0.5
        '''
        unread = ct.readability(self.text, lang='chinese')
        return unread['readability3']
        
       

    def lexical_richness(self,) -> list:
        '''
        信息密度
        指标实词比例：实词*/总词数
        '''
        content_pos ={'n', 'v', 'a', 'ad', 'vd', 'an','nz'}
        words = pseg.lcut(self.cntext)
        
        total_words = 0
        content_words = 0
        
        for word, flag in words:
            # 排除标点符号（根据词性过滤）
            if flag == 'x':
                continue
            total_words += 1
            if flag in content_pos:
                content_words += 1
        
        # 计算实词比例
        if total_words == 0:
            return 0.0
        return round(content_words / total_words, 3)
    def sentiment_score(self):
        sents = [SnowNLP(sent).sentiments for sent in self.text.split('。') if sent.strip()]
        
        return SnowNLP(self.text).sentiments,np.var(sents)  # 0~1，越接近1越积极
    # 使用 moving average TTR
    
        # 词汇丰富度，用TF-IDF进行加权的moving average TTR，TF-IDF score高的唯一词更加重要
    def weighted_ttr(self, allow_pos=('n', 'vn', 'v','nz','an'), topK=30):
        # Step 1: 分词并过滤指定词性
        words_pos = pseg.lcut(self.cntext)
        filtered_words = [word for word, pos in words_pos if pos in allow_pos]
        total_words = len(filtered_words)
        if total_words == 0:
            return 0.0
        
        # Step 2: 提取唯一词及其 TF-IDF 权重
        keywords_with_weight = jieba.analyse.extract_tags(
            self.cntext, 
            topK=topK, 
            withWeight=True,
            allowPOS=allow_pos
        )
        
        # 构建词到权重的映射（去重）
        word_weight = {word: weight for word, weight in keywords_with_weight}
        
        # Step 3: 计算加权 TTR
        unique_words = set(filtered_words)
        weighted_sum = sum(word_weight.get(word, 0) for word in unique_words)
        
        return round(weighted_sum / total_words, 4)

    def moving_ttr(self,window=20):
        words = [w for w in jieba.lcut(self.cntext) if self.is_valid_keyword(w)]
        if len(words) < window:
            return 0.0
        ttrs = []
        for i in range(len(words) - window + 1):
            window_words = words[i:i+window]
            unique = len(set(window_words))
            ttrs.append(unique / window)
        return sum(ttrs) / len(ttrs)
    # 感染力/唤醒度评估
    def sentiment_by_weight(self,ChineseEmoBank_df):

        ret = ct.sentiment_by_weight(self.text, ChineseEmoBank_df['ChineseEmoBank'],  ['valence', 'arousal'],
                        lang = 'chinese')
        print(ret)
        Valence = round(ret['valence']/ret['word_num'],4)

        Arousal = round(ret['arousal']/ret['word_num'],4)
        # print('Valence:',Valence,' Arousal:',Arousal)
        return Valence,Arousal
    
    def keyword_coverage(self,
            keyword_weights: dict,  # 格式: {'关键词': 权重}
            mode: str = 'weighted'  # 可选 'binary'(原始方法) 或 'weighted'(加权)
        ) -> float:
        """
        计算文本对关键词列表的覆盖率（支持加权模式）
        
        Args:
            text: 待评估的文本
            keyword_weights: 关键词及其权重（如TF-IDF值、历史互动系数）
            mode: 计算模式 ('binary' 或 'weighted')
        
        Returns:
            覆盖率得分 (0~1)
        """
        if not keyword_weights:
            return 0.0
        
        # 预处理：文本分词并转换为集合（加速查找）
        words = self.extract_keywords_with_tag_split()  # 中文分词
        
        total_weight = 0.0
        matched_weight = 0.0
        
        for keyword, weight in keyword_weights.items():
            # 判断关键词是否存在于文本中
            if keyword in words:  # 或使用更复杂的匹配逻辑（如子串匹配）
                matched_weight += weight if mode == 'weighted' else 1.0
            total_weight += weight if mode == 'weighted' else 1.0
        
        return round(matched_weight / total_weight,4) if total_weight != 0 else 0.0
    def unify_eval(self, keyword_weights,ChineseEmoBank_df):
        readability = round(10 / self.unreadability(),4)
        lexical_richness = self.lexical_richness()
        sentiment_by_weight = self.sentiment_by_weight(ChineseEmoBank_df)[1]
        keyword_coverage = self.keyword_coverage(keyword_weights)
        return {'可读性':readability, '信息密度':lexical_richness, '感染力':sentiment_by_weight, '流行契合度':keyword_coverage}
        
        
    def is_valid_keyword(self,word):
        '''
        单词中包含至少一个中文字符（Unicode 范围为 \u4e00-\u9fa5）。
        单词不是仅由数字、非字母数字字符（包括标点符号和下划线 _）组成。
        单词去除前后空白字符后的长度大于 1
        单词不在停用词列表中
        '''
        return (
            re.search(r'[\u4e00-\u9fa5]', word)
            and not re.fullmatch(r'[\d\W_]+', word)
            and len(word.strip()) > 1
            and word not in self.stopwords
        )
    # 从正文中提取关键词，标签里的内容也做分词
    def extract_keywords_with_tag_split(self, topk=100):
        
        tags = re.findall(r"#(.*?)#", self.text)

        # 分词标签
        tag_words = []
        for tag in tags:
            tag = re.sub(r"[^\u4e00-\u9fa5]", "", tag)
            tag_words.extend([w for w in jieba.analyse.extract_tags(tag, topK=3) if self.is_valid_keyword(w)])

        # 去除标签文本
        clean_text = re.sub(r"#.*?#", "", self.text)
        raw_keywords = jieba.analyse.extract_tags(clean_text, topK=topk) #  allowPOS=('n', 'vn', 'v')
        keywords = [kw for kw in raw_keywords if self.is_valid_keyword(kw)]
        keywords.extend(tag_words)
        return list(keywords)

def read_file_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 使用 '##' 分割文件内容
    sections = content.split('##')
    
    # 创建一个空字典来存储结果
    result_dict = {}
    
    for section in sections[1:]:
        lines = section.strip().split('\n')
        if lines:
            # 第一行是字典的键
            key = lines[0].strip()
            # 剩余的行是文案内容
            content = '\n'.join(lines[1:]).strip()
            result_dict[key] = content
    
    return result_dict

if __name__ == '__main__':
    #%%
    file_path = "all_content_comp.txt"
    
    text_dict = read_file_to_dict(file_path)
    # print(text_dict)
    #%%
    ChineseEmoBank_df = ct.load_pkl_dict('ChineseEmoBank.pkl')
    freq_weight = pd.read_csv('关键词频次.csv')
    weight_dict = {row['关键词']: row['weight'] for index, row in freq_weight.iterrows()}
    
    for k, text in text_dict.items():
        print(k)
        evaler = MultiAspect_eval(text)
        print(evaler.unify_eval(weight_dict, ChineseEmoBank_df))
        # 其他评估方法的调用...


    
    