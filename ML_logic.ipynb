{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "27ebb4df-d424-4347-a47c-a5c64de60f3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#importing necessary libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import ast"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "16ca9e49-bb28-4211-bec3-427a38e5f9ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# reading datasets and choosing nessasry features for learning and removing null and duplicate values\n",
    "df1 = pd.read_csv('tmdb_5000_credits.csv')\n",
    "df2 = pd.read_csv('tmdb_5000_movies.csv')\n",
    "main = df1.merge(df2,on ='title')\n",
    "main = main[['movie_id','title','overview','genres','keywords','cast','crew']]\n",
    "df2.isnull().sum()\n",
    "main.dropna(inplace = True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "a4c1fca3-2d71-4b03-84aa-6e8cf4fac87f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#helper function to convert the list of dict to a list of str with only names\n",
    "#helper function to pick onlt the top 3 ACTORS\n",
    "#helper function to pick only the director\n",
    "\n",
    "\n",
    "def convert(obj):\n",
    "    l = []\n",
    "    for i in ast.literal_eval(obj):\n",
    "        l.append(str(i['name']))\n",
    "    return l\n",
    "\n",
    "def character(obj):\n",
    "    l = []\n",
    "    counter = 0 #only need top 3 ACTORS NAME\n",
    "    for i in ast.literal_eval(obj):\n",
    "        if counter !=3:\n",
    "            l.append(str(i['name']))\n",
    "            counter+=1\n",
    "        else:\n",
    "            break\n",
    "    return l\n",
    "\n",
    "def director(obj):\n",
    "    l = []\n",
    "    for i in ast.literal_eval(obj):\n",
    "        if i['job'] == \"Director\":\n",
    "                l.append(str(i['name']))\n",
    "                break\n",
    "    return l"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "c334d22f-ec10-40f8-9831-746708ba26da",
   "metadata": {},
   "outputs": [],
   "source": [
    "#applying the fuction to each attribute\n",
    "main['genres'] = main['genres'].apply(convert)\n",
    "main['keywords'] = main['keywords'].apply(convert)\n",
    "main['cast'] = main['cast'].apply(character)\n",
    "main['crew'] = main['crew'].apply(director)\n",
    "main['overview'] = main['overview'].apply(lambda x:x.split())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "cccf2a8f-ed5c-41a7-8f0a-e5b10f609a43",
   "metadata": {},
   "outputs": [],
   "source": [
    "#to remove space in order to maintain uniqeness of values\n",
    "main['genres'] = main['genres'].apply(lambda x :[i.replace(\" \",\"\")for i in x])\n",
    "main['keywords'] = main['keywords'].apply(lambda x :[i.replace(\" \",\"\")for i in x])\n",
    "main['cast'] = main['cast'].apply(lambda x :[i.replace(\" \",\"\")for i in x])\n",
    "main['crew'] = main['crew'].apply(lambda x :[i.replace(\" \",\"\")for i in x])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "f1908ca9-9817-4da7-8fea-9aa45ef90d39",
   "metadata": {},
   "outputs": [],
   "source": [
    "#concatonating all the attributes ibto a single attribute called the tags\n",
    "main['tags'] = main['genres']+main['keywords']+main['cast']+main['overview']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "ef29ebfe-fef7-4d24-ab66-5293fe0cd28b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\viswa\\AppData\\Local\\Temp\\ipykernel_8644\\3648744551.py:5: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  movies['tags'] = movies['tags'].apply(lambda x: \" \".join(x))\n",
      "C:\\Users\\viswa\\AppData\\Local\\Temp\\ipykernel_8644\\3648744551.py:6: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  movies['tags'] = movies['tags'].apply(lambda x: x.lower())\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>movie_id</th>\n",
       "      <th>title</th>\n",
       "      <th>tags</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>19995</td>\n",
       "      <td>Avatar</td>\n",
       "      <td>action adventure fantasy sciencefiction cultur...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>285</td>\n",
       "      <td>Pirates of the Caribbean: At World's End</td>\n",
       "      <td>adventure fantasy action ocean drugabuse exoti...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>206647</td>\n",
       "      <td>Spectre</td>\n",
       "      <td>action adventure crime spy basedonnovel secret...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>49026</td>\n",
       "      <td>The Dark Knight Rises</td>\n",
       "      <td>action crime drama thriller dccomics crimefigh...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>49529</td>\n",
       "      <td>John Carter</td>\n",
       "      <td>action adventure sciencefiction basedonnovel m...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   movie_id                                     title  \\\n",
       "0     19995                                    Avatar   \n",
       "1       285  Pirates of the Caribbean: At World's End   \n",
       "2    206647                                   Spectre   \n",
       "3     49026                     The Dark Knight Rises   \n",
       "4     49529                               John Carter   \n",
       "\n",
       "                                                tags  \n",
       "0  action adventure fantasy sciencefiction cultur...  \n",
       "1  adventure fantasy action ocean drugabuse exoti...  \n",
       "2  action adventure crime spy basedonnovel secret...  \n",
       "3  action crime drama thriller dccomics crimefigh...  \n",
       "4  action adventure sciencefiction basedonnovel m...  "
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#make a new dataframe with the necessary features\n",
    "#making all the tags a string\n",
    "#change all the tag content to lowercase\n",
    "movies = main[['movie_id','title','tags']]\n",
    "movies['tags'] = movies['tags'].apply(lambda x: \" \".join(x))\n",
    "movies['tags'] = movies['tags'].apply(lambda x: x.lower())\n",
    "movies.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "3c609772-0cb1-4670-8e83-bfcc3bcc4d13",
   "metadata": {},
   "outputs": [],
   "source": [
    "#from nltk import porterStemmer in order to combine similar sounding words like love lovong loved etc to love\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "ps = PorterStemmer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "573b6ac8-1515-4780-9c8e-d97ca2efd274",
   "metadata": {},
   "outputs": [],
   "source": [
    "#helper funtion to combine similar sounding words like love lovong loved etc to love\n",
    "def stem(text):\n",
    "    y = []\n",
    "    for i in text.split():\n",
    "        y.append(ps.stem(i))\n",
    "    return \" \".join(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "4a35d75e-e68d-4066-97e7-2e8eb6d8c67d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\viswa\\AppData\\Local\\Temp\\ipykernel_8644\\1255633842.py:2: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame.\n",
      "Try using .loc[row_indexer,col_indexer] = value instead\n",
      "\n",
      "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  movies[\"tags\"]=movies[\"tags\"].apply(stem)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'action adventur fantasi sciencefict cultureclash futur spacewar spacecoloni societi spacetravel futurist romanc space alien tribe alienplanet cgi marin soldier battl loveaffair antiwar powerrel mindandsoul 3d samworthington zoesaldana sigourneyweav in the 22nd century, a parapleg marin is dispatch to the moon pandora on a uniqu mission, but becom torn between follow order and protect an alien civilization.'"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#call the function to amke the changes\n",
    "movies[\"tags\"]=movies[\"tags\"].apply(stem)\n",
    "movies.tags[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "617623c0-c741-44e4-bd1c-13820169fe21",
   "metadata": {},
   "outputs": [],
   "source": [
    "#import scikitlearn to use contvercorizer in order to choose thw most used 5000 words in all of the tags combined by removing\n",
    "#the stops words from the english language\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "cv = CountVectorizer(max_features = 5000,stop_words = 'english')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "953cdf8c-141c-4591-a085-43f822ccc938",
   "metadata": {},
   "outputs": [],
   "source": [
    "#convert the features into an array\n",
    "vectors = cv.fit_transform(movies['tags']).toarray()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "6ddf523b-211e-4eba-887a-025ab0e9c50a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['000', '007', '10', ..., 'zone', 'zoo', 'zooeydeschanel'],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#print the features\n",
    "cv.get_feature_names_out()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "0c8a8df8-d5ef-400e-8713-a2bb568b846e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1.        , 0.08471737, 0.08740748, ..., 0.04499213, 0.        ,\n",
       "        0.        ],\n",
       "       [0.08471737, 1.        , 0.06253054, ..., 0.02414023, 0.        ,\n",
       "        0.02654659],\n",
       "       [0.08740748, 0.06253054, 1.        , ..., 0.02490677, 0.        ,\n",
       "        0.        ],\n",
       "       ...,\n",
       "       [0.04499213, 0.02414023, 0.02490677, ..., 1.        , 0.03962144,\n",
       "        0.04229549],\n",
       "       [0.        , 0.        , 0.        , ..., 0.03962144, 1.        ,\n",
       "        0.08714204],\n",
       "       [0.        , 0.02654659, 0.        , ..., 0.04229549, 0.08714204,\n",
       "        1.        ]])"
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#distance of each movie with every other movie by using cosine_similarity as ot parameter\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "similarity = cosine_similarity(vectors)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "e122f842-42f5-47e2-8687-e819741bb9f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "#this line is used t+in the cell below the main use is to assign an index to each movie so afer it is sorted it dosn't lose its position\n",
    "#later it is sorted based on the second attribute and we return oly top 5 values\n",
    "#sorted(list(enumerate(similarity[0])) ,reverse = True,key = lambda x:x[1])[1:6]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 115,
   "id": "547dd3bd-78d4-49e3-93e3-4d198ee3ac69",
   "metadata": {},
   "outputs": [],
   "source": [
    "#helper function that takes the name of the movie and return the index of the most similar \n",
    "def recom(movie):\n",
    "    movie_index = movies[movies['title'] == movie].index[0]\n",
    "    distances = similarity[movie_index]\n",
    "    movie_list = sorted(list(enumerate(distances)) ,reverse = True,key = lambda x:x[1])[1:6]\n",
    "\n",
    "    for i in movie_list:\n",
    "        #return the name of the movie based on its index\n",
    "        print(movies.iloc[i[0]].title)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "6215b9ca-2c68-4f44-819a-e306b694268f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The Dark Knight\n",
      "Batman\n",
      "Batman\n",
      "10th & Wolf\n",
      "The Dark Knight Rises\n"
     ]
    }
   ],
   "source": [
    "recom('Batman Begins')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "4005bfd4-c47d-467c-a64c-c9eee161961c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#send the filer to pycram to display it to the website\n",
    "import pickle\n",
    "pickle.dump(movies.to_dict(),open('movies_dict.pkl','wb'))\n",
    "pickle.dump(similarity,open('similarity.pkl','wb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8e09b4d3-ad6a-4355-9c12-7eb475ace3a4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
