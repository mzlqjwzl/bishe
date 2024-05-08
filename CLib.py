#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import os
import json
import keras
from keras.applications.resnet import ResNet50,preprocess_input
from keras.models import Model,Sequential
from keras.preprocessing import image
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.layers import Input,Dense,Lambda,LSTM,GRU,Bidirectional,RepeatVector,ReLU,Flatten,TimeDistributed,Softmax,Embedding,Concatenate
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
from matplotlib import pyplot as plt 
import h5py
import cv2


# ## 序列化 文本编码

class Tokenizer():
    def __init__(self):
        self.maxlen = 0
        self.index_to_words = dict()
        self.word_index = dict()
        self.word_index['<start>'] = 0
        self.word_index['<end>'] = 1
        self.word_index['<unk>'] = 2
    def fit_on_texts(self,texts):
        for text in texts:
            for word in text:
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)
        self.maxlen = max([len(text) for text in texts])
        self.maxlen =  50 if self.maxlen > 50 else self.maxlen
        self.index_to_words = {_id:word for word,_id in self.word_index.items()}

    def texts_to_sequences(self,texts):
        seqs = list()
        for text in texts: 
            seq = [0]
            for word in text:
                seq.append(self.word_index.get(word, 2))
            seq = seq + [1] * (self.maxlen + 2 - len(seq))
            if len(seq)>self.maxlen+2:
                seq = seq[:self.maxlen+2]
            seqs.append(seq)
        return np.array(seqs)
    
    def sequence_to_text(self,sequence):
        return ''.join([self.index_to_words[index] for index in sequence])


def preprocessing(datas:dict(),tokenizer=None,maxlen=None):
    texts = [v['caption'] for k,v in datas.items()]
    filenames = np.array([v['filename'] for k,v in datas.items()])
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
#     if maxlen is None:
#         maxlen = max([len(sentence) for sentence in sequences])
#     sequences = pad_sequences(sequences,maxlen=maxlen,padding='post')
#     sequences = np.expand_dims(sequences,axis=2)
    return sequences,tokenizer,filenames

# ##  加载图像

def load_imgs(image_dir:str,filenames):
    imgs = list()
    for filename in filenames:
        img = load_img(os.path.join(image_dir,filename), target_size=(244, 244))
        img = image.img_to_array(img)
        imgs.append(img)
    imgs = np.stack(imgs)
    return imgs.astype(np.int32)


# ### 特征提取模型:ResNet50

def get_preprocess_model():
    input_tensor = Input((244, 244, 3))
    input_tensor = Lambda(preprocess_input)(input_tensor)
    base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', pooling='max', include_top=False)
    output = base_model.output
    return Model(base_model.input, output)

# ## 对训练数据进行特征提取，并生成hdf5数据文件

def save_preprocess(image_dir:str, filenames, model,savename:str):
    batch_size = 16
    filenames = set(filenames)
    filenames = list(filenames)
    print('处理%d条数据' % len(filenames))
    with h5py.File(savename, 'a') as f:
        while len(filenames) > 0:
            print('\r剩余%d' % len(filenames), end=' ')
            batch_filenames = filenames[-batch_size:]
            for _ in range(batch_size):
                if len(filenames) > 0:
                    filenames.pop()
            imgs = load_imgs(image_dir, batch_filenames)
            features = model.predict(imgs)
            for filename, feature in zip(batch_filenames, features):
                f.create_dataset(filename,data=feature)



# 加载特征
def load_features(preprocess_filename:str, imagefilenames):
    with h5py.File(preprocess_filename, "r") as f:
        features = list()
        for filename in imagefilenames:
            features.append(np.array(f[filename]))
        features = np.array(features)
        return features
            


# ## 数据生成器
# 为训练模型提供数据生成器；
# 注：不训练模型不要运行

def generator(preprocess_filename:str,filenames,sequences,batch_size=32):
    assert len(filenames) == len(sequences)
    buffer = 1
    while True:
        batch_index = np.random.choice(len(filenames),size=batch_size,replace=False)
        batch_filenames = filenames[batch_index]
        batch_sequences = sequences[batch_index]
        features = load_features(preprocess_filename, batch_filenames)
        y_seqs = np.expand_dims(batch_sequences,axis=2)
        yield [features, batch_sequences[:,:-1]], y_seqs


# ## 定义模型

def caption_model(output_sequence_length:int,vocab_size:int,hidden_size:int=128,embed_size:int=128):
    def expand_dims_layer(x,axis=-1):
        return keras.backend.expand_dims(x, axis=axis)
    def argmax_layer(x,axis=-1):
        return keras.backend.argmax(x, axis=axis) 
    def expand_dims_layer(x,axis=-1):
        return keras.backend.expand_dims(x, axis=axis)

    input_features = Input((2048,))
    features = Dense(embed_size)(input_features)
    features = Lambda(expand_dims_layer,arguments={'axis':1})(features)
    input_seqs = Input((output_sequence_length-1,))
    embed_layer = Embedding(vocab_size,embed_size)
    embeds = embed_layer(input_seqs)
    lstm_inputs = Concatenate(axis=1)([features, embeds])
    lstm_layer = LSTM(hidden_size, return_sequences=True, return_state=True, dropout=0.2)
    lstm_output,_,_ = lstm_layer(lstm_inputs)
    dense_layer = Dense(vocab_size)
    output = TimeDistributed(dense_layer)(lstm_output)
    output = Softmax()(output)
    train_model = Model([input_features, input_seqs],output)
    
    
    lstm_inputs = features
    initial_state = None
    output_seqs = list()
    for i in range(output_sequence_length):
        print('\r', i, end='')
        lstm_output,state_h,state_c = lstm_layer(lstm_inputs,initial_state)
        lstm_output = Flatten()(lstm_output)
        output = dense_layer(lstm_output)
        word_index = Lambda(argmax_layer,arguments={'axis':1})(output)
        embeds = embed_layer(word_index)
        lstm_inputs = Lambda(expand_dims_layer,arguments={'axis':1})(embeds)
        initial_state = (state_h,state_c)
        output_seqs.append(word_index)
    predict_model = Model(input_features,output_seqs)
    return train_model,predict_model

# 格式化输出结果
cleantxt = lambda x:x.replace('<start>','').replace('<end>','')

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    from PIL import Image, ImageDraw, ImageFont
    # 判断是否OpenCV图片类型,自动转换
    #print(type(img))
    #print(isinstance(img, numpy.ndarray))
    #if (isinstance(img, numpy.ndarray)):  
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    # fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    fontStyle = ImageFont.truetype("/System/Library/Fonts/STHeiti Light.ttc", textSize, encoding="utf-8")
    # 绘制文本
    # 4. 在图片上写字
    # 第一个参数：指定文字区域的左上角在图片上的位置(x,y)
    # 第二个参数：文字内容
    # 第三个参数：字体
    # 第四个参数：颜色RGB值
    #img_draw.text((chars_x, chars_y), chars, font=ttf, fill=(255,0,0))
    draw.text((left, top), text, fill=textColor, font=fontStyle) #font=fontStyle, , fill=textColor
    # 转换回OpenCV格式
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    #image = np.asarray(img)
    return image


if __name__ == '__main__':
    pass

