# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 15:11:15 2019

@author: LABA
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from keras.models import Model, load_model
from keras import  backend as K
import numpy as np

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred- y_true)))


def train_test_split_fuc(data):
        
    train, tt = train_test_split(data, test_size=0.3)
    
    vlidation, test = train_test_split(tt, test_size=0.5)

        
    return train, vlidation, test

ratings = pd.read_csv('ratings.csv')

user_unique = ratings['userId'].unique()
movie_unque = ratings['movieId'].unique()

rate_max = ratings['rating'].max()
rate_min = ratings['rating'].min()

label  = LabelEncoder()
ratings.userId = pd.DataFrame({'userId':ratings['userId'].values}).apply(label.fit_transform)
ratings.movieId = pd.DataFrame({'movieId':ratings['movieId'].values}).apply(label.fit_transform)
ratings = shuffle(ratings)

train, validation, test = train_test_split_fuc(ratings)

n_users = len(ratings.userId.unique())
n_movies = len(ratings.movieId.unique())

movie_input = Input(shape=(1,), name = 'Movie-Input')
movie_embedding = Embedding(n_movies+1, 100, name = 'Movie-Embedding')(movie_input)
movie_vec = Flatten(name='Flatten-Movies')(movie_embedding)

user_input = Input(shape=(1,), name = "User-Input")
user_embedding = Embedding(n_users+1, 100, name = 'User-Embedding')(user_input)
user_vec = Flatten(name = 'Flatten-User')(user_embedding)

prod = Dot(name = 'Dot-Product', axes=1)([movie_vec, user_vec])
model = Model([user_input ,movie_input], prod)
model.compile(loss = rmse, optimizer = 'rmsprop' ,metrics=['acc'])
#model.summary()

#history = model.fit([train.userId, train.movieId], train.rating, 
#                    validation_data=([validation.userId, validation.movieId], validation.rating),
#                    batch_size=64, 
#                    epochs=10, 
#                    verbose=1)


model = load_model('CF.h5', custom_objects={'rmse': rmse})

movie_data = np.array(list(set(ratings.movieId)))[0:4]
user_data = np.array(list(14 for i in range(len(movie_data))))


predictions = model.predict([user_data, movie_data])
predictions = np.array([a[0] for a in predictions])
recommended_movie_ids = (-predictions).argsort()[:]
print(recommended_movie_ids)
print(predictions[recommended_movie_ids])



