from flask import Flask, render_template, request
import pickle
import numpy as np
from collections import defaultdict
import pandas as pd
import csv
# import surprise
from matrix_1k import matrix_model, user_recommendations, movie_neighbors
app = Flask(__name__)

movies_DNN = pd.read_csv('../data/movies.dat', sep='::', engine='python', encoding='latin-1', names=['movieId', 'title', 'genres'])

### Matrix_Movie_1M start from here
with open('case_insensitive_movies_list.pkl', 'rb') as f:

    case_insensitive_movies_list = pickle.load(f)

with open('unique_movies.pkl', 'rb') as f:
    unique_movies = pickle.load(f)

with open('U.pkl', 'rb') as f:
    U = pickle.load(f)

with open('S.pkl', 'rb') as f:
    S = pickle.load(f)

with open('V.pkl', 'rb') as f:
    V = pickle.load(f)

with open('movies_dict.pkl', 'rb') as f:
    movies_dict = pickle.load(f)

def get_user_recommendations(user_id, number_of_results):
    # 确保打开文件时使用了正确的文件路径
    with open('user_recommendations.csv') as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        data = {}

        for row in csvreader:
            data.setdefault(row[0], []).append(row)

    recommendations = data.get(user_id, [])[:number_of_results]
    return recommendations


def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id 
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    cosine_similarity_values = similarity[sort_indexes[:top_n]]  
    print(cosine_similarity_values)
    
    return sort_indexes[:top_n], cosine_similarity_values
    
def get_similar_movies(movie_name, top_n, k=50):
    sliced = V.T[:, :k] 
    movie_id = movies_dict[movie_name]
    indexes, scores = top_cosine_similarity(sliced, movie_id, top_n)
    
    # 初始化结果字符串
    result_str = f"Top {top_n} movies similar to {movie_name} are:\n\n"
    
    # 填充结果字符串
    for i in range(1, len(indexes)):  # 跳过第一个索引，因为它是查询的电影本身
        movie_title = unique_movies[indexes[i]]
        score = scores[i-1]  # 分数数组的索引从0开始
        result_str += f"{i}. {movie_title}, score: {score:.4f}\n"
    
    return result_str


def get_possible_movies(movie):

    temp = ''
    possible_movies = case_insensitive_movies_list.copy()
    for i in movie :
      out = []
      temp += i
      for j in possible_movies:
        if temp in j:
          out.append(j)
      if len(out) == 0:
          return possible_movies
      out.sort()
      possible_movies = out.copy()

    return possible_movies

class invalid(Exception):
    pass

def recommender(user_input):
    
    try:
      movie_name = user_input
      movie_name_lower = movie_name.lower()
      if movie_name_lower not in case_insensitive_movies_list :
        raise invalid
      else :
        # movies_list[case_insensitive_country_names.index(movie_name_lower)]
        num_recom = 10
        result_1 = get_similar_movies(unique_movies[case_insensitive_movies_list.index(movie_name_lower)], num_recom)
        return result_1

    except invalid:

      possible_movies = get_possible_movies(movie_name_lower)

      if len(possible_movies) == len(unique_movies) :
        return "movie no exist"
      else :
        suggestions = [unique_movies[case_insensitive_movies_list.index(i)] for i in possible_movies]
        # 使用 <pre> 标签来保持换行和空格，以及缩进来改善输出格式
        suggestions_str = "Entered Movie name is not matching with any movie from the dataset . Please check the below suggestions\n\n"
        for i, suggestion in enumerate(suggestions, 1):  # 从1开始计数
            suggestions_str += f"    {i}. {suggestion}\n"
        return suggestions_str
### Matrix_Movie_1M End


### Matrix_User_1M Start from here
with open('top_n.pkl', 'rb') as f:
    top_n = pickle.load(f)

with open('df_movie.pkl', 'rb') as f:
    df_movie = pickle.load(f)

def giverUsr(userid):
    userid = int(userid)
    movie_str = ''  
    for uid, user_ratings in top_n.items():
        if uid == userid:
            movie_titles = []
            for i, (iid, _) in enumerate(user_ratings, start=1):  
                movie_title = df_movie[df_movie['movieId'] == iid]['title'].values
                if len(movie_title) > 0:
                    movie_titles.append(f"{i}. {movie_title[0]}")  

            #
            movie_str = '\n'.join(movie_titles)
            movie_str = '\n' + movie_str if movie_str else ''  
    return movie_str
### Matrix_User_1M End

### Matrix_100k Start
Matrix_100k_model = matrix_model()
def recommender_100k(movie_name):
    movie_list = movie_neighbors(Matrix_100k_model , movie_name)
    print(movie_list)
    if movie_list == []:
        return "No movie with that title"
    movie_str = '\n'.join(movie_list)
    movie_str = '\n' + movie_str if movie_str else ''  
    return movie_str
    
def giverUsr_100k(userid):
    try:
        userid = int(userid)
    except ValueError:
        return "Please input integer for userID"
    movie_list = user_recommendations(Matrix_100k_model , id=int(userid))
    movie_str = '\n'.join(movie_list)
    movie_str = '\n' + movie_str if movie_str else ''  
    return movie_str
### Matrix_100k End

### knn preparing start
movieId2Idx = {movies_DNN.movieId[i]: i for i in range(len(movies_DNN))}

overviewsPath = 'overview.csv'
overview = pd.read_csv(overviewsPath)
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = overview["overview"].fillna("")
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(corpus)
from sklearn.metrics.pairwise import linear_kernel
knnCosineSims = linear_kernel(vectors, vectors)

with open('topRatedIdx.pickle', 'rb') as f:
    topRatedIdx = pickle.load(f)
with open('watchedIdx.pickle', 'rb') as f:
    watchedIdx = pickle.load(f)
### knn preparing end

### kNN_User Start
def meanSim(idxList, cosineSims):
    sim = np.zeros(cosineSims[0].shape)
    for idx in idxList:
        sim += cosineSims[idx]
    sim /= len(idxList)
    return sim

def knnUser(userId, k=10):
    try:
        userId = int(userId)
    except ValueError:
        return f"Please enter an integer between 1 and 6040."
    
    if userId < 1 or userId > 6040:
        return f"Please enter an integer between 1 and 6040."
    
    ret = f"Recommending for userId: {userId}.\n\n"

    topRatedSim = meanSim(topRatedIdx[userId-1], knnCosineSims)
    similarIdx = (-topRatedSim).argsort()

    recommendIdxList = []
    for idx in similarIdx:
        if idx in watchedIdx[userId-1]:
            continue
        recommendIdxList.append(idx)
        if len(recommendIdxList) == k:
            break

    ret += "\n".join(f"{i+1}. {movies_DNN.title[idx]}" for i, idx in enumerate(recommendIdxList))
    return ret
### kNN_User End

### kNN_Movie Start
def knnMovie(movieIdx, k=10):
    try:
        movieIdx = int(movieIdx)
    except ValueError:
        return f"Please enter a movie index between 0 and {len(movies_DNN)-1}."
    
    if movieIdx < 0 or movieIdx >= len(movies_DNN):
        return f"Please enter a movie index between 0 and {len(movies_DNN)-1}."
    
    return recommend(movieIdx, k)

def recommend(targetIdx, k=10):
    ret = f'Recommending movies that are similar to "{movies_DNN.title[targetIdx]}":\n\n'

    cosineSim = knnCosineSims[targetIdx]
    similarMovies = (-cosineSim).argsort()[1:k+1]
    ret += "\n".join(f'{i+1}. {movies_DNN.title[idx]}' for i, idx in enumerate(similarMovies))
    return ret
### kNN_Movie End

### kNN_UserMovie Start
def knnUserMovie(user_input, k=10, userWeight=0.9):
    inputs = user_input.split(';')
    errorMsg = f'Input error. Usage: <User ID>;<Movie Index>.\nUser ID ranges from 1 to 6040 and Movie Index ranges from 0 to {len(movies_DNN)-1}.'
    if len(inputs)!=2:
        return errorMsg
    try:
        userId = int(inputs[0])
        movieIdx = int(inputs[1])
    except ValueError:
        return errorMsg
    if userId < 1 or userId > 6040:
        return f"User ID error. Please enter an integer between 1 and 6040."
    if movieIdx < 0 or movieIdx >= len(movies_DNN):
        return f"Movie index error. Please enter a movie index between 0 and {len(movies_DNN)-1}."
    
    topRatedSim = meanSim(topRatedIdx[userId-1], knnCosineSims)
    movieSim = knnCosineSims[movieIdx]
    sims = userWeight * topRatedSim + (1-userWeight) * movieSim
    similarIdx = (-sims).argsort()

    ret = f'Recommending for user {userId} with the movie "{movies_DNN.title[movieIdx]}":\n\n'

    recommendIdxList = []
    for idx in similarIdx:
        if idx in watchedIdx[userId-1]:
            continue
        recommendIdxList.append(idx)
        if len(recommendIdxList) == k:
            break

    ret += "\n".join(f"{i+1}. {movies_DNN.title[idx]}" for i, idx in enumerate(recommendIdxList))
    return ret
### kNN_UserMovie End

### DNN_User Start
def get_user_recommendations_DNN(user_id):
    number_of_results =10
    with open('user_recommedations.csv', encoding='utf-8') as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        data = {}

        for row in csvreader:
            data.setdefault(row[0], []).append(row)

    recommendations = data.get(user_id, [])[:int(number_of_results)]
    
    # 格式化输出
    result = "\n"
    for i, rec in enumerate(recommendations, start=1):
        movie_name = rec[1]  # 假设电影名称是每条记录的第二个元素
        movie_year = rec[2]  # 假设电影年份是每条记录的第三个元素
        result += f"{i}. {movie_name} ({movie_year})\n"
    
    return result
### DNN_User End

### DNN_Movie Start
def get_movie_recommendations_DNN(movie_id):
    """
    根据电影ID获取推荐电影列表。
    """
    number_of_results=10
    recommendations_list = []
    with open('similar_movies.csv', encoding="utf8") as file:
        csvreader = csv.reader(file, delimiter=',', quotechar='"')
        dataMovie = {}
        for i, row in enumerate(csvreader):
            if i % 21 == 0:  # 假设每21行是一个新的电影分组
                key = row[0]
            dataMovie.setdefault(key, []).append(row)

    recommendations = dataMovie.get(str(movie_id), [])[:number_of_results]
    for rec in recommendations:
        # 注意：这里假设推荐列表中包含电影ID，名称和相似度分数
        movie_details = movies_DNN[movies_DNN['movieId'] == int(rec[0])].iloc[0]
        recommendations_list.append((movie_details['title'], "Unknown Year", rec[3]))  # 年份未知
    return recommendations_list

def recommend_movies_DNN(movie_name):
    """
    根据电影名称提供推荐或建议。
    """
    number_of_results=10
    # 尝试精确匹配
    exact_match = movies_DNN[movies_DNN['title'].str.lower() == movie_name.lower()]
    if not exact_match.empty:
        movie_id = exact_match.iloc[0]['movieId']
        recommendations = get_movie_recommendations_DNN(movie_id)
        result_str = f"Result: Top {number_of_results} movies similar to '{exact_match.iloc[0]['title']}' are:\n\n"
        result_str += "\n".join(f"{i+1}. {title}, score: {score}" for i, (title, year, score) in enumerate(recommendations))
        return result_str

    # 没有精确匹配，尝试模糊匹配提供建议
    possible_matches = movies_DNN[movies_DNN['title'].str.lower().str.contains(movie_name.lower())]
    if possible_matches.empty:
        return "Movie does not exist. Please try another name."
    else:
        suggestions = "Entered movie name does not match exactly. Please check the below suggestions:\n\n"
        suggestions += "\n".join(f"{i+1}. {match}" for i, match in enumerate(possible_matches['title'], 1))
        return suggestions
### DNN_Movie End

def k_nearest_neighbors(user_type, user_input):
    # 实现细节...
    return f"K-Nearest Neighbors for {user_type} with input: {user_input}"

def deep_neural_networks(user_type, user_input):
    # 实现细节...
    return f"Deep Neural Networks for {user_type} with input: {user_input}"

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""  # 用于存储处理结果
    user_input = "" 
    algorithm = ""
    user_type = ""
    
    if request.method == "POST":
        algorithm = request.form.get("algorithm")
        user_type = request.form.get("input_type")
        user_input = request.form.get("user_input")

        # 根据用户选择的算法调用对应的函数
        if algorithm == "Matrix Factorization 1M" and user_type == "Movie":
            result = recommender(user_input)
        elif algorithm == "Matrix Factorization 1M" and user_type == "User":
            result = giverUsr(user_input)      
        elif algorithm == "Matrix Factorization 100k" and user_type == "Movie":
            result = recommender_100k(user_input)
        elif algorithm == "Matrix Factorization 100k" and user_type == "User":
            result = giverUsr_100k(user_input)           
        elif algorithm == "K-Nearest Neighbors" and user_type == "Movie":
            result = knnMovie(user_input)
        elif algorithm == "K-Nearest Neighbors" and user_type == "User":
            result = knnUser(user_input)
        elif algorithm == "K-Nearest Neighbors" and user_type == "Both":
            result = knnUserMovie(user_input)
        elif algorithm == "Deep Neural Networks" and user_type == "User":
            result = get_user_recommendations_DNN(user_input)
        elif algorithm == "Deep Neural Networks" and user_type == "Movie":
            result = recommend_movies_DNN(user_input)

    return render_template("index.html", result=result, user_input=user_input, algorithm=algorithm, user_type=user_type)

if __name__ == "__main__":
    app.run(debug=True)
