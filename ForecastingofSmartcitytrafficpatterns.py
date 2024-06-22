#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.initializers import he_normal
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_rows', 10)


# In[ ]:


import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)


# In[ ]:


df_train = pd.read_csv('./data/train_aWnotuB.csv', parse_dates=[0], infer_datetime_format=True)
df_train


# In[ ]:


train = df_train.pivot(index='DateTime',columns='Junction', values='Vehicles')
train


# In[ ]:


train = train.fillna(0)


# In[ ]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[ ]:


scaler = MinMaxScaler(feature_range=(0, 1))


# In[ ]:


train[train.columns] = scaler.fit_transform(train[train.columns])


# In[ ]:


train


# In[ ]:


nb_forecast_per_junction = 24 * (31 + 31 + 30 + 31) # Days in jul + aug + sep + oct


# In[ ]:


num_feats = 4
seq_len = 24 * 2 
num_outputs = 4
num_hidden = 4 
bs = 128
epochs = 500
LOG_PATH = "checkpoints/" + time.strftime("%Y-%m-%d_%H%M-")+"s2s-concat-Conv1d.hdf5"


# In[ ]:


def make_input_seqs(data, seq_len, train_split=0.9):
    seq_len = seq_len + 1
    result = []
    for index in range(len(data) - seq_len):
        result.append(data[index: index + seq_len, :])
    result = np.array(result) 
    train_ind = round(train_split * result.shape[0])
    train = result[:int(train_ind), :, :]
    x_train = train[:, :-1, :]
    y_train = train[:, -1, :]
    x_test = result[int(train_ind):, :-1, :]
    y_test = result[int(train_ind):, -1, :]

    return [x_train, y_train, x_test, y_test]


# In[ ]:


X_train, y_train, X_test, y_test = make_input_seqs(train.values, seq_len)


# In[ ]:


import keras.backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 


# In[ ]:


from keras.optimizers import adam, rmsprop, sgd


# In[ ]:


adam = adam(clipvalue=0.5)
rmsprop = rmsprop(lr = 0.005, decay = 0.05)

optim = adam


# In[ ]:


from keras.layers import *
from keras.layers.core import *
from keras.layers.recurrent import *
from keras.models import *
from keras.activations import *


# In[ ]:


def attention_n_days_ago(inputs, days_ago)
    time_steps = days_ago * 24
    suffix = str(days_ago) +'_days'
    
    a = Permute((2, 1),
                name='Attn_Permute1_' + suffix)(inputs)
    a = Dense(time_steps,
              activation='softmax',
              name='Attn_DenseClf_' + suffix)(a)
    
    
    feats_depth = int(inputs.shape[2])
    avg = Lambda(lambda x: K.expand_dims(x, axis = 1),
                 name='Attn_Unsqueeze_' + suffix)(inputs)
    avg = SeparableConv2D(feats_depth, (1,1),
                          name='Attn_DepthConv_' + suffix)(avg)
    avg = Lambda(lambda x: K.squeeze(x, 1),
                 name='Attn_Squeeze_'+ str(days_ago) + '_days')(avg)
    
    
    a_probs = Permute((2, 1),
                      name='Attn_Permute1_' + suffix)(avg)
    
    out = Concatenate(name='Attn_cat_'+ suffix)([inputs, a_probs])
    return out


# In[ ]:


def Net(num_feats, seq_len, num_hidden, num_outputs):
    x = Input(shape=(seq_len, num_feats))

    
    enc = CuDNNGRU(seq_len,
                   return_sequences=True,
                   stateful = False,
                   name = 'Encoder_RNN')(x)
    
    
    attention_0d = attention_n_days_ago(enc, 0)
    attention_1d = attention_n_days_ago(enc, 1)
    attention_2d = attention_n_days_ago(enc, 2)
    attention_4d = attention_n_days_ago(enc, 4)
    attention_1w = attention_n_days_ago(enc, 7)
    attention_2w = attention_n_days_ago(enc, 14)
    attention_1m = attention_n_days_ago(enc, 30)
    attention_2m = attention_n_days_ago(enc, 60)
    attention_1q = attention_n_days_ago(enc, 92)
    attention_6m = attention_n_days_ago(enc, 184)
    attention_3q = attention_n_days_ago(enc, 276)
    attention_1y = attention_n_days_ago(enc, 365)
    
    att = Concatenate(name='attns_cat', axis = 1)([attention_0d,
                                                   attention_1d,
                                                   attention_2d,
                                                   attention_4d,
                                                   attention_1w,
                                                   attention_2w,
                                                   attention_1m,
                                                   attention_2m,
                                                   attention_1q,
                                                   attention_6m,
                                                   attention_3q,
                                                   attention_1y])
    
    
        
    att = Dense(seq_len, activation=None, name='Dense_merge_attns')(att)

    dec = CuDNNGRU(num_hidden,
                   return_sequences=False,
                   stateful = False,
                   name='Decoder_RNN')(att)
    out = Dense(num_outputs, activation=None,
                name = 'Classifier')(dec) 
    
    model = Model(inputs=x, outputs=out)
                          
    model.compile(loss= root_mean_squared_error, optimizer = optim)
    return model


# In[ ]:


model = Net(num_feats, seq_len, num_hidden, num_outputs)


# In[ ]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# In[ ]:


from keras.callbacks import History, ModelCheckpoint, CSVLogger, EarlyStopping


# In[ ]:


history = History()
checkpointer = ModelCheckpoint(filepath= LOG_PATH,
                               verbose=1, save_best_only=False)
csv_logger = CSVLogger("checkpoints/" + time.strftime("%Y-%m-%d_%H%M-")+'training.log')
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')


# In[ ]:


model.fit(X_train, y_train,
          batch_size=bs,
          epochs=epochs,
          validation_split=0.05,
          shuffle=False,
          callbacks=[history,checkpointer,csv_logger,early_stop])


# In[ ]:


def get_states(model):
    return [K.get_value(s) for s,_ in model.state_updates]

def set_states(model, states):
    for (d,_), s in zip(model.state_updates, states):
        K.set_value(d, s)


# In[ ]:


RNNs_states = get_states(model)


# In[ ]:


def plot_preds(y_truth, y_pred):
    for junction in range(4):
        plt.figure
        plt.plot(y_truth[:,junction], color = 'blue', label = 'Real traffic')
        plt.plot(y_pred[:,junction], color = 'orange', label = 'Predicted traffic')
        plt.title('Traffic Forecasting at junction %i' % (junction+1))
        plt.xlabel('Number of hours from Start')
        plt.ylabel('Traffic')
        plt.legend()
        plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




