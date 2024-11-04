import streamlit as st
import pickle
import pandas as pd

def recomm(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recom_movies =[]
    for i in movie_list:
        movie_id = i[0]
        recom_movies.append(movies.iloc[i[0]].title)
    return recom_movies

movies_dict = pickle.load(open('movies_dict.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))
movies = pd.DataFrame(movies_dict)
st.title('Movie Recommendation system')
selected_movie_name = st.selectbox('Enter the movie name ! ',movies['title'].values)
if st.button('Recommend'):
    recommendations = recomm(selected_movie_name)
    st.write(f'Movies similar to {selected_movie_name} are : ')
    for i in recommendations:
        st.write(i)