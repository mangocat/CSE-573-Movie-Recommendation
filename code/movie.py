import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 确保使用的是TensorFlow 2.x
print(tf.__version__)

# 加载数据
ratings = pd.read_csv("../data/ratings.csv")
movies = pd.read_csv('../data/movies.csv')
tags = pd.read_csv('../data/tags.csv')
genome_scores = pd.read_csv('../data/genome-scores.csv')
genome_tags = pd.read_csv('../data/genome-tags.csv')

# 假设这些DataFrame已经按照您的需求进行了适当的处理

# 定义模型
class RecommenderModel(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(RecommenderModel, self).__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_dim, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.movie_bias = tf.keras.layers.Embedding(num_movies, 1)
        
    def call(self, inputs):
        user_vec = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vec = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        
        dot_user_movie = tf.reduce_sum(user_vec * movie_vec, axis=1)
        x = dot_user_movie + user_bias[:, 0] + movie_bias[:, 0]
        
        return tf.nn.sigmoid(x) * (max_rating - min_rating) + min_rating

# 准备数据
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()
min_rating = min(ratings['rating'])
max_rating = max(ratings['rating'])

# 创建训练和测试数据
X = ratings[['userId', 'movieId']].values
# 将用户ID和电影ID调整为从0开始的索引
X[:, 0] = pd.factorize(X[:, 0])[0]
X[:, 1] = pd.factorize(X[:, 1])[0]
y = ratings['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values  # 归一化评分
train_indices, test_indices, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 模型实例化和编译
embedding_dim = 50
model = RecommenderModel(num_users, num_movies, embedding_dim)
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=0.001))

# 训练模型
model.fit(x=train_indices, y=y_train, batch_size=32, epochs=5, verbose=1, validation_data=(test_indices, y_test))

# 评估模型
model.evaluate(test_indices, y_test)
