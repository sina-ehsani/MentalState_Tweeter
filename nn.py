import argparse
import zipfile
import sklearn.metrics
import pandas as pd

import json
import re
import h5py
import numpy as np
import pandas as pd

import keras


from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout, Masking, Embedding, SimpleRNN, Bidirectional, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import jaccard_similarity_score, jaccard_score


emotions = ["anger", "anticipation", "disgust", "fear", "joy", "love",
            "optimism", "pessimism", "sadness", "surprise", "trust"]
emotion_to_int = {"0": 0, "1": 1, "NONE": -1}

def read_tabled():

  

  read_csv_kwargs = dict(sep="\t",
                         converters={e: emotion_to_int.get for e in emotions})
  train_data = pd.read_csv("2018-E-c-En-train.txt", **read_csv_kwargs)
  dev_data = pd.read_csv("2018-E-c-En-dev.txt", **read_csv_kwargs)
  test_real_data = pd.read_csv("2018-E-c-En-test.txt", **read_csv_kwargs)
  return(train_data,dev_data,test_real_data)


def pre_train(train_data,dev_data,test_real_data):

  tokenize = re.compile(r"\d+|[^\d\W]+|\S").findall
  df_tokenized = [tokenize(comment) for comment in train_data["Tweet"]]

  index_to_token = ['unk'] + [''] + sorted(set(tokens
                                     for tweets in df_tokenized
                                     for tokens in tweets))
  token_to_index = {c: i for i, c in enumerate(index_to_token)}
  max_tokens = max(len(tweets)
                   for tweets in df_tokenized)

  #Read Embedding
  embeddings_index = {}
  with open('data/glove.twitter.27B.200d.txt') as f:
      for line in f:
          word, coefs = line.split(maxsplit=1)
          coefs = np.fromstring(coefs, 'f', sep=' ')
          embeddings_index[word] = coefs

  #TRAINING DATA:
          
  with h5py.File('data/tweet-emotions.hdf5', 'w') as f:
      f.attrs["vocabulary"] = json.dumps(index_to_token)
      matrix_in = np.zeros(shape=(len(df_tokenized), max_tokens))
      
      EMBEDDING_DIM=200
      embedding_matrix = np.zeros((len(index_to_token), EMBEDDING_DIM))
      for  i , word in enumerate(index_to_token):
          embedding_vector = embeddings_index.get(word)
          if embedding_vector is not None:
              # words not found in embedding index will be all-zeros.
              embedding_matrix[i] = embedding_vector
          else: embedding_matrix[i] = embeddings_index.get('unk')
          
      for i, tweet in enumerate(df_tokenized):
          for j , token in enumerate(tweet):
              matrix_in[i,j] = token_to_index[token]
      print(matrix_in.shape)
      matrix_out = train_data.loc[:,'anger':'trust'].values
      train = f.create_group("train")
      train.create_dataset("input", compression="gzip", data=matrix_in)
      train.create_dataset("output", compression="gzip", data=matrix_out)
      train.create_dataset("embedding_matrix", compression="gzip", data=embedding_matrix)
      
  #DEV DATA:
  df_tokenized = [tokenize(comment) for comment in dev_data["Tweet"]]

  with h5py.File('data/tweet-emotions-dev.hdf5', 'w') as f:
      matrix_in = np.zeros(shape=(len(df_tokenized), max_tokens))
                
      for i, tweet in enumerate(df_tokenized):
          for j , token in enumerate(tweet):
              if token in token_to_index:
                  matrix_in[i,j] = token_to_index[token]
              else: matrix_in[i,j] = token_to_index['unk']
              
      print(matrix_in.shape)
      matrix_out = dev_data.loc[:,'anger':'trust'].values
      test = f.create_group("test")
      test.create_dataset("input", compression="gzip", data=matrix_in)
      test.create_dataset("output", compression="gzip", data=matrix_out)
      
  #Real test DATA: 
  df_tokenized = [tokenize(comment) for comment in test_real_data["Tweet"]]   

  with h5py.File('data/tweet-emotions-test.hdf5', 'w') as f:
      matrix_in = np.zeros(shape=(len(df_tokenized), max_tokens))
          
      for i, tweet in enumerate(df_tokenized):
          for j , token in enumerate(tweet):
              if token in token_to_index:
                  matrix_in[i,j] = token_to_index[token]
              else: matrix_in[i,j] = token_to_index['unk']
              
      print(matrix_in.shape)
      real_test = f.create_group("real_test")
      real_test.create_dataset("input", compression="gzip", data=matrix_in)

  return()  


def training():

  with h5py.File('data/tweet-emotions.hdf5', 'r') as f:
      vocabulary = json.loads(f.attrs["vocabulary"])
      train = f["train"]
      train_in = np.array(train["input"])
      train_out = np.array(train["output"])
      embedding_matrix = np.array(train["embedding_matrix"])
  with h5py.File('data/tweet-emotions-dev.hdf5', 'r') as f:
  #     vocabulary = json.loads(f.attrs["vocabulary"])
      test = f["test"]
      test_in = np.array(test["input"])
      test_out = np.array(test["output"])


def model(EMBEDDING_DIM= 200,max_tokens = 85):

  embedding_layer = Embedding(len(vocabulary),
                              EMBEDDING_DIM,
                              weights=[embedding_matrix],
                              input_length=max_tokens,
                              trainable=False,
                              mask_zero=True)
  model=Sequential()
  model.add(embedding_layer)
  model.add(Bidirectional(GRU(16, return_sequences=False, dropout=0.3)))
  # model.add(SeqSelfAttention())
  model.add(Dense(train_out.shape[1], activation='sigmoid'))
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


  es = EarlyStopping(monitor='val_loss', verbose=1 ,   patience=10) # Findes early stopping using for the loss on the validation dataset, adding a delay of 3 to the trigger in terms of the number of epochs on which we would like to see no improvement. 
  mc = ModelCheckpoint('best_model_1layerGRU.h5', monitor='val_loss', verbose=1 ,save_best_only=True) #The callback will save the model to file, the one with best overall performance.

  model.summary()
  return(model, es,mc)

def train(model, es,mc):
  with h5py.File('data/tweet-emotions.hdf5', 'r') as f:
      vocabulary = json.loads(f.attrs["vocabulary"])
      train = f["train"]
      train_in = np.array(train["input"])
      train_out = np.array(train["output"])
      embedding_matrix = np.array(train["embedding_matrix"])
  with h5py.File('data/tweet-emotions-dev.hdf5', 'r') as f:
  #     vocabulary = json.loads(f.attrs["vocabulary"])
      test = f["test"]
      test_in = np.array(test["input"])
      test_out = np.array(test["output"])

  model.fit(x=train_in,y=train_out,batch_size=64,epochs=50, callbacks=[es,mc],verbose=1,validation_data=(test_in,test_out))

  # Lets Combine the dev and train data, and use the dev for train as well ( 5 epocs):
  train_test_in = np.vstack((train_in, test_in))
  train_test_out = np.vstack((train_out, test_out))

  import tensorflow as tf
  del model
  model = tf.keras.models.load_model('best_model_1layerGRU.h5')
  print(model.summary())

  es2 = EarlyStopping(monitor='val_loss', verbose=1 ,   patience=2 , baseline=0.33156) # Findes early stopping using for the loss on the validation dataset, adding a delay of 10 to the trigger in terms of the number of epochs on which we would like to see no improvement. 
  mc2 = ModelCheckpoint('best_model_1layerGRU_alldata.h5', monitor='val_loss', verbose=1 ,save_best_only=True) #The callback will save the model to file, the one with best overall performance.
  model.fit(x=train_test_in,y=train_test_out,batch_size=64,epochs=5, callbacks=[es2,mc2],verbose=1,validation_data=(test_in,test_out))


  y_pred = model.predict(train_test_in)

  #Fint the best treshhold:
  trsh_list2=[]
  for j in range(11):
    temp=y_pred.copy()
    temp[temp>.33]=1
    temp[temp<=.33]=0

    for i in range(len(y_pred)):
      temp[i][j]=y_pred[i][j]

    from sklearn.metrics import jaccard_score
    best_score=0
    best_tr=0
    for i in range(100):
      rnd=np.random.rand()
      temp1=temp.copy()
      temp1[temp1>rnd]=1
      temp1[temp1<=rnd]=0
      jac_score=jaccard_score(train_test_out,temp1,average='samples')
      if jac_score > best_score:
        best_score=jac_score
        best_tr=rnd

    print(j)
    print(best_score)
    print(best_tr)
    trsh_list2.append(best_tr)

  with h5py.File('data/tweet-emotions-test.hdf5', 'r') as f:
      real_test = f["real_test"]
      test_real_in = np.array(real_test["input"])

  y_pred = model.predict(test_real_in)
  temp5=y_pred.copy()

  for i,l in enumerate(temp5):
    for k , j in enumerate(l):
      # print(k)
      if j>trsh_list2[k]:
        temp5[i][k]=int(1)
      else:
        temp5[i][k]=int(0)

  print("Jaccard score",jaccard_score(train_test_out,temp2,average='samples'))

  read_csv_kwargs = dict(sep="\t",
                         converters={e: emotion_to_int.get for e in emotions})

  test_real_data = pd.read_csv("2018-E-c-En-test.txt", **read_csv_kwargs)
  test_real_data.loc[:,'anger':'trust']=temp5.astype(int)
  test_real_data.to_csv("E-C_en_pred.txt", index=False,  sep="\t" , header=True)

  return()





