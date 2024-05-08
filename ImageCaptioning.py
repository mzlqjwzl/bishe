#!/usr/bin/env python
# coding: utf-8

# # 图像字幕大师
# 
# 本项目会通过keras搭建一个深度卷积神经网络来为图片生成一行字幕

# ## 数据集
# 采用数据集：MS COCO dataset
# 下载地址：http://cocodataset.org/

import sys
from CLib import *

# ##  读取数据
print('正在读取数据...')

train_dir = r'images/train2014/'
val_dir = r'images/val2014/'
test_dir = r'images/test2014/'
train_datas = json.load(open('train.json'))
val_datas = json.load(open('val.json'))

'''
print(json.dumps(list(train_datas.items())[:3], ensure_ascii=False,indent=4))
print(json.dumps(list(val_datas.items())[:3], ensure_ascii=False,indent=4))
'''

# 加载训练数据，并得到序列化对象

train_datas = json.load(open('train.json'))
train_tokenized, tokenizer, train_filenames = preprocessing(train_datas)
# print(len(tokenizer.word_index))

val_tokenized,_,val_filenames = preprocessing(val_datas,
                                              tokenizer=tokenizer,
                                              maxlen=train_tokenized.shape[1])


'''
print(list(train_datas.items())[:2])
print(list(val_datas.items())[:2])
print(train_tokenized.shape)
print(val_tokenized.shape)
print(list(tokenizer.word_index.items())[:6])
print(train_tokenized[1])
print(val_tokenized[1])
'''

preprocess_model = get_preprocess_model()

def create_hdf5():
    ## 对训练数据进行特征提取，并生成hdf5数据文件
    save_preprocess(train_dir, train_filenames, preprocess_model,'train_preprocess.hdf5')
    print('训练集处理完成')
    save_preprocess(val_dir, val_filenames, preprocess_model,'val_preprocess.hdf5')
    print('验证集处理完成')


# ## 定义模型
print('正在创建模型...')
vocab_size = len(tokenizer.word_index) # 3346
output_sequence_length = train_tokenized.shape[1] #52
#train_model, predict_model = caption_model(train_tokenized.shape[1], len(tokenizer.word_index), hidden_size=512,embed_size=512)
train_model, predict_model = caption_model(output_sequence_length, vocab_size, hidden_size=512,embed_size=512)
print('\r模型准备完成。')

# train_model.save('weight/model.h5')
'''
train_model.summary()
predict_model.summary()
'''

# ## 加载模型
#model_file = 'weight/42/ep071val_loss0.433val_acc0.865.h5'
model_file = 'weight/ep071val_loss0.433val_acc0.865.h5'
train_model.load_weights(model_file)
print('模型权重已加载。')


# ## 训练模型
def train_model():
    # 生成数据用于训练
    batch_size = 16
    train_gen = generator('train_preprocess.hdf5', train_filenames, train_tokenized,batch_size=batch_size)
    val_gen = generator('val_preprocess.hdf5', val_filenames, val_tokenized,batch_size=batch_size)
    for (features,x_seqs),y_seqs in train_gen:
        print(features.shape)
        print(x_seqs.shape)
        print(y_seqs.shape)
        print(tokenizer.sequence_to_text(x_seqs[0]))
        break

    # 定义训练模型的参数
    lr = 0.0001
    train_model.compile(loss = sparse_categorical_crossentropy, 
                  optimizer = Adam(lr), 
                  metrics = ['accuracy'])
    mc = ModelCheckpoint(filepath=r'weight\ep{epoch:03d}val_loss{val_loss:.3f}val_acc{val_accuracy:.3f}.h5',
                         save_best_only=True, save_weights_only=True, verbose=1)
    es = EarlyStopping(monitor='loss',patience=4, min_delta=0.001, verbose=1)
    his = train_model.fit_generator(generator=train_gen,
                        steps_per_epoch=6000,
                        epochs=1000,
                        callbacks=[mc,es],
                        validation_data=val_gen,
                        validation_steps=300)


    # ##  画出训练曲线图像
    # 训练完成后画出训练曲线图
    begin = 10
    loss=his.history['loss'][begin:]
    val_loss=his.history['val_loss'][begin:]
    plt.plot(range(begin+1 , begin + 1 + len(loss)),loss,color='red')
    plt.plot(range(begin+1 , begin + 1 + len(val_loss)),val_loss,color='blue')
    plt.legend(labels=['loss','val_loss',])


    # ## 验证集预测
    # 在验证集上进行预测，对比结果；


    imgs = load_imgs(val_dir ,val_filenames[:10])
    features = load_features('val_preprocess.hdf5', val_filenames[:10])
    texts = val_tokenized[:10]
    predicts = predict_model.predict(features)
    predicts = np.array(predicts)
    predicts = np.transpose(predicts,axes=(1,0))

    for predict,img,text in zip(predicts,imgs,texts) :
        print('预测:',tokenizer.sequence_to_text(predict))
        print('真实:',tokenizer.sequence_to_text(text))
        plt.imshow(img)
        plt.show()


    # ##  测试集数据验证

    # 测试结果
    test_dir = r'images/test2014/'
    test_filenames = os.listdir(test_dir)
    # 随机选5张
    indexs = np.random.choice(len(test_filenames), size=5, replace=False)
    imgs = load_imgs(test_dir ,np.array(test_filenames)[indexs])
    features = preprocess_model.predict(imgs)
    predicts = predict_model.predict(features)
    predicts = np.array(predicts)
    predicts = np.transpose(predicts,axes=(1,0))
    for predict,img in zip(predicts,imgs):
        print(tokenizer.sequence_to_text(predict))
        plt.imshow(img)
        plt.show()


# ## 实际应用：预测图像
# 把待预测的图像放到 'test'目录下，批量进行预测

# 批量预测图像调用方法 
def predict_image(imgs):
    features = preprocess_model.predict(imgs)
    predicts = predict_model.predict(features)
    predicts = np.array(predicts)
    predicts = np.transpose(predicts,axes=(1,0))
    pred_txts = [cleantxt(tokenizer.sequence_to_text(predict)) for predict in predicts]
    return pred_txts


def predict_test():

    test_dir = r'test/'
    test_filenames = os.listdir(test_dir)
    #indexs = np.random.choice(len(test_filenames), size=5, replace=False)
    imgs = load_imgs(test_dir, np.array(test_filenames))
    print('正在预测图像...')
    pred_txts = predict_image(imgs)
    for pred_txt,img in zip(pred_txts,imgs):
        print('-'*40)
        plt.imshow(img.astype(np.uint8))
        plt.show()
        print('预测结果:',pred_txt)


def predict_video():
    vfile = 'video/757244476-1-16.mp4'
    
    Video = cv2.VideoCapture(vfile)
    Video.set(1, 1)
    counter = 0

    print('视频文件:%s' % vfile)
    Frame = 1
    text = ''
    while 1:
        #print('-'*40)
        # 逐帧读取
        ret, frame = Video.read()
        if ret:
            # 判断是否要加字幕
            if Frame % 600 == 50 or text=='':
                Img = frame.copy()
                Img = cv2.resize(Img, (244,244))
                print(frame.shape, Img.shape)
                #break;
                # 识别图片
                pred_txts = predict_image(np.array([Img]))
                text = '字幕君:%s，当前帧：%d'% (pred_txts, Frame)
                print(text)
                #break;
            # 标字幕
            x, y = 20, frame.shape[0] - 50
            frame = cv2ImgAddText(frame, text, x, y , textColor=(255,255,255), textSize=20)
   
            cv2.imshow('Video', frame)
            key = cv2.waitKey(5)
            if key != -1: exit()

            Frame += 1

        else:
            break

predict_video()
#sys.exit()

