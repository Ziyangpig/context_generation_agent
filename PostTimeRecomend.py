'''
author: Ziyang Zhu
date: 2024-04-22
'''
import os
os.environ["OMP_NUM_THREADS"] = '1'
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import json


# =================================
# 定义高互动微博 & 计算特征
# =================================
class PostTimeRecomend:
    def __init__(self,df):
       
        self.df = df
        self.X = None
        
        if self.df is None:
            self.df = self.fake_data()
        self.max_k = min(self.df.shape[0]//2,7)


    def preprocess(self,df):
        # 计算互动指数（按用户定义的权重）
        df['interaction_score'] = df['likes']*0.4 + df['shares']*0.3 + df['comments']*0.3

        # 筛选高互动微博（取前30%）
        threshold = df['interaction_score'].quantile(0.7)
        high_interact_df = df[df['interaction_score'] >= threshold].copy()

        # 特征工程
        # 将星期转换为One-Hot编码
        weekday_dummies = pd.get_dummies(high_interact_df['weekday'], prefix='weekday')
        high_interact_df = pd.concat([high_interact_df, weekday_dummies], axis=1)
        # print(weekday_dummies.columns)
        # 标准化互动分数（用于聚类权重）
        scaler = StandardScaler()
        high_interact_df['scaled_score'] = scaler.fit_transform(high_interact_df[['interaction_score']])

        # 构造特征矩阵X
        features = ['hour','scaled_score'] + list(weekday_dummies.columns)
        self.X = high_interact_df[features]
        return high_interact_df

    # ==============================
    # 使用K-means进行聚类
    # ==============================

    def auto_select_k(self):
        """
        自动选择最佳K值
        :param X: 标准化后的特征矩阵
        :param max_k: 最大尝试K值
        :return: 最佳K值
        """

        
        metrics = {
            'silhouette': {'best_score': -1, 'best_k': 2},
            'calinski': {'best_score': -1, 'best_k': 2},
            'davies': {'best_score': float('inf'), 'best_k': 2}
        }

        for k in range(2, self.max_k+1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.X)
            # print('k=',k)
            # 轮廓系数（越大越好）
            sil_score = silhouette_score(self.X, labels)
            if sil_score > metrics['silhouette']['best_score']:
                metrics['silhouette'] = {'best_score': sil_score, 'best_k': k}
                
            
            # Calinski-Harabasz（越大越好）
            ch_score = calinski_harabasz_score(self.X, labels)
            if ch_score > metrics['calinski']['best_score']:
                metrics['calinski'] = {'best_score': ch_score, 'best_k': k}
                
            
            # Davies-Bouldin（越小越好）
            db_score = davies_bouldin_score(self.X, labels)
            if db_score < metrics['davies']['best_score']:
                metrics['davies'] = {'best_score': db_score, 'best_k': k}
            # print('silhouette_score:',sil_score,'\n',
            #     'calinski_harabasz_score:',ch_score,'\n',
            #     'davies_bouldin_score:',db_score,'\n')
        # 投票选择最终K值
        k_counts = Counter([
            metrics['silhouette']['best_k'],
            metrics['calinski']['best_k'],
            metrics['davies']['best_k']
        ])
        return metrics,metrics['calinski']['best_k']


    def elbow_method(self):
        # 使用肘部法可视化判断k选择是否合适
        sse = []
        k_range = range(2, self.max_k+1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.X)
            sse.append(kmeans.inertia_)

        # 绘制肘部图
        plt.figure(figsize=(10,6))
        plt.plot(k_range, sse, 'bx-')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('Elbow Method For Optimal K')
        plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
        # plt.show()


    # ==============================
    # 可视化分析结果
    # ==============================
    def res_info(self,high_interact_df,):
        # 创建时段列
        high_interact_df['period'] = (high_interact_df['hour'] // 3) * 3
        plt.figure(figsize=(12, 6))
        pivot_table = high_interact_df.pivot_table(
            index='period',
            columns='weekday',
            values='cluster',
            aggfunc='count',
            fill_value=0
        )

        # 重命名星期标签
        weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        pivot_table.columns = weekday_labels

        # 绘制热力图
        sns.heatmap(
            pivot_table,
            cmap="YlGnBu",
            linewidths=0.3,
            annot=True,
            fmt="d",
            annot_kws={"size": 8}
        )

        plt.title('High Interaction Clusters Distribution by Period and Weekday',fontsize=12, fontweight='bold')
        plt.xlabel('Weekday')
        plt.ylabel('Period of Day')
        plt.savefig('heatmap_period.png', dpi=300, bbox_inches='tight')
        # plt.show()

        # ======================================
        # 步骤5：解读聚类结果
        # ======================================

        # 分析每个簇的黄金时段
        # cluster_summary = high_interact_df.groupby('cluster').agg({
        #     'hour': ['mean', lambda x: x.mode()[0]],
        #     'weekday': lambda x: x.mode()[0],
        #     'content_type': lambda x: x.mode()[0]
        # }).reset_index()

        # cluster_summary.columns = ['Cluster', 'Avg Hour', 'Mode Hour', 'Mode Weekday', 'Dominant Content Type']
        cluster_summary = high_interact_df.groupby('cluster').agg({
            'hour': ['mean', lambda x: x.mode()[0]],
            'weekday': lambda x: x.mode()[0],
            
        }).reset_index()

        cluster_summary.columns = ['Cluster', 'Avg Hour', 'Mode Hour', 'Mode Weekday']
        print("\n聚类中心特征分析:")
        print(cluster_summary)
        return cluster_summary

    def fake_data(self):
        # 随机生成
        np.random.seed(42) 

        # 生成1000条模拟数据
        n_samples = 1000
        data = {
            # 发布时间的小时（0-23），模拟晚间高概率
            'date': pd.date_range(start='2024-12-01', periods=n_samples, freq='D'),
            'hour': np.random.choice(np.arange(24), p=[
                0.01,0.01,0.0,0.0,0.01,0.01,  # 0-5时0.04
                0.02,0.02,0.03,0.03,0.02,0.03,  # 6-11时0.15
                0.06,0.05,0.03,0.02,0.02,0.03, # 12-17时0.21
                0.10,0.12,0.15,0.10,0.08,0.05  # 18-23时0.6
            ], size=n_samples),
            
            # 星期（0=周一, 6=周日），模拟周末高概率
            'weekday': np.random.choice([0,1,2,3,4,5,6], p=[
                0.12,0.12,0.12,0.12,0.12,  # 周一至周五
                0.20,0.20                   # 周六、周日
            ], size=n_samples),
            
            # 内容类型（0=皮肤，1=赛事，2=其他）
            'content_type': np.random.choice([0,1,2], p=[0.4,0.3,0.3], size=n_samples),
            
            # 互动数据（点赞、转发、评论）
            'likes': np.random.poisson(5000, n_samples),
            'shares': np.random.poisson(300, n_samples),
            'comments': np.random.poisson(800, n_samples)
        }

        df = pd.DataFrame(data)
        return df

    def post_time_recom_main(self):
        if self.df is None:
            self.df = self.fake_data()
        high_interact_df = self.preprocess(self.df)
        #TODO: 动态更新 保留半年数据，最新时间传递给爬虫开始爬取时间 
        
        # 投票机制选择K值
        metrecs, optimal_k = self.auto_select_k()
        print(f"投票机制选择K值:\n{metrecs}")
        print(f"矩阵投票选择 最佳K值为：{optimal_k}")
        # 肘部法可视化辅助
        self.elbow_method()
        # 使用最优K值训练模型
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        high_interact_df['cluster'] = kmeans.fit_predict(self.X)
        
        cluster_summary = self.res_info(high_interact_df)
        plt.show()
        with open('res.txt', 'w') as f:
            f.write('投票机制选择K值:\n')
            f.write(json.dumps(metrecs))
            f.write("聚类中心特征分析:\n")
            f.write(cluster_summary.to_string(index=False))

if __name__ == "__main__":
    PostTimeRecomend(None).post_time_recom_main()
    
        
    

