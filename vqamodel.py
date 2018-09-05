import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten,Input, LSTM, Embedding, Dense,TimeDistributed,Dropout
from keras.models import Model, Sequential, load_model
from keras.preprocessing import image,sequence
from keras.preprocessing.text import Tokenizer
from nltk.probability import FreqDist
from keras.utils import to_categorical
import os
import pandas as pd
import numpy as np
import glob
import cv2



def save_key_image(path, save_path, papers=10, img_size = 224):
    name = os.path.splitext(os.path.split(path)[-1])[0]
    print(name)
    new_path = save_path + name+'/'
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    cap = cv2.VideoCapture(path) #读入视频文件
    frames_num=cap.get(7) #获得总帧数
    for i in np.linspace(0, frames_num-1, num = papers, dtype = int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        rval, frame=cap.read()
        frame=cv2.resize(frame,(img_size, img_size),fx=0,fy=0,interpolation=cv2.INTER_AREA)
        cv2.imwrite(new_path + str(i) + '.jpg',frame) #存储为图像
    cap.release()

#Saving key images from videos, default 10 frames and image size is 224
def save_key_images(path, save_path, papers=10 , img_size = 224):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    list(map(lambda video : save_key_image(video, save_path, papers, img_size),glob.glob(path+'*.mp4')))

#Video model
def video_model_create(img_size,video_input):
    vision_model = Sequential()
    vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(img_size, img_size, 3)))
    # vision_model.add(Conv2D(64, (3, 3), activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Dropout(0.25))
    # vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # vision_model.add(Conv2D(128, (3, 3), activation='relu'))
    # vision_model.add(MaxPooling2D((2, 2)))
    # vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # vision_model.add(Conv2D(256, (3, 3), activation='relu'))
    # vision_model.add(Conv2D(256, (3, 3), activation='relu'))
    # vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Flatten())

    encoded_frame_sequence = TimeDistributed(vision_model)(video_input) 
    encoded_video = LSTM(32)(encoded_frame_sequence)
    # encoded_video = Dropout(0.25)(encoded_video)
    return encoded_video 

#Question model
def encoded_video_question_create(video_question_input):
    embedded_question = Embedding(input_dim=2000, output_dim=64, input_length=15)(video_question_input)
    encoded_question = LSTM(32)(embedded_question)
    encoded_question = Dropout(0.25)(encoded_question)
    question_encoder = Model(inputs=video_question_input, outputs=encoded_question)

    encoded_video_question = question_encoder(video_question_input)
    return encoded_video_question

#Aggregative model
def vqa_model_create(papers, img_size):
    video_input = Input(shape=(papers, img_size, img_size, 3))
    video_question_input = Input(shape=(15,), dtype='int32')

    encoded_video = video_model_create(img_size,video_input)
    encoded_video_question = encoded_video_question_create(video_question_input)

    merged = keras.layers.concatenate([encoded_video, encoded_video_question])
    output = Dense(1000, activation='softmax')(merged)
    vqa_model = Model(inputs=[video_input, video_question_input], outputs=output)    
    
    #compile model 
    vqa_model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])
    vqa_model.summary()
    return vqa_model


#Loading txt into dataframe including questions and answers
def txt_to_df(path):
    df = pd.read_csv(path, header=None, dtype = object)
    df.columns = ['name'] + [str(j)+str(i) for i in ['1','2','3','4','5'] for j in ['q','a1','a2','a3']]
    file_names = df['name']
    df = pd.wide_to_long(df, stubnames=['q','a1','a2','a3'],i='name',j='qa')
    df['index'] = list(map(lambda x :x[0],df.index))
    df['qa'] = list(map(lambda x :x[1],df.index))
    df['index'] = df['index'].astype('category')
    df['index'].cat.set_categories(file_names,inplace = True)
    df.sort_values(['index','qa'],ascending = True,inplace = True)
    df[['a1','a2','a3']] = df[['a1','a2','a3']].applymap(lambda x: x.replace('-',' '))
    return df,file_names

#Transforming answers into input formats
def answer_to_input(df_a):
    # ans_list = sorted(map(lambda word : word[0],FreqDist(df_a['a1'].append(df_a['a2']).append(df_a['a3'])).most_common(1000)))
    ans_list = sorted(map(lambda word : word[0],FreqDist(df_a['a1']).most_common(1000)))
    #pd.DataFrame(ans_list).to_csv('temp.csv',header=None,index=None)
    
#    df_a[['a1','a2','a3']] = df_a[['a1','a2','a3']].applymap(lambda x: x if x in ans_list else '0')
#    df_a['lable'] = df_a['a1']+','+df_a['a2']+','+df_a['a3']    
    answer_input = df_a['a1'].str.get_dummies(sep = ',')[ans_list].values
    return answer_input, ans_list

#Transforming questions into input formats   
def question_to_input(df_q1,df_q2):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_q1 + df_q2)
    encoded_1 = tokenizer.texts_to_sequences(df_q1)
    encoded_2 = tokenizer.texts_to_sequences(df_q2)
    question_input_train = sequence.pad_sequences(encoded_1, maxlen=15)
    question_input_test = sequence.pad_sequences(encoded_2, maxlen=15)

    return question_input_train,question_input_test

#Transforming images into input formats 
def image_to_input(img_path, image_names, papers, img_size, copies):
    x_img = np.zeros((len(image_names)*copies, papers, img_size, img_size, 3))
    for num,image_name in enumerate(image_names):
        for index,name in enumerate(os.listdir(img_path + image_name)):
            img = image.load_img(img_path + image_name+'/'+ name)
            for i in range(copies):
                x_img[copies*num + i][index] = image.img_to_array(img)
    return x_img / 255

#main
if __name__ == '__main__': 
    #parameter
    img_size = 40
    papers = 10
    copies = 5
    epochs = 80
    batch_size = 128

    #save
    data_save  = False
    model_save = False
    model_load = False
    
#data    
    #save images
    img_path_train = './image_train_' + str(img_size) + '_' + str(papers)+ '/'  
    img_path_test  = './image_test_'  + str(img_size) + '_' + str(papers)+ '/'
    if not os.path.exists(img_path_train):
        path_train = "./VQADatasetA_20180815/train/"        
        save_key_images(path_train, img_path_train, papers = papers, img_size = img_size)  
    if not os.path.exists(img_path_test):
        path_test  = "./VQADatasetA_20180815/test/"
        save_key_images(path_test, img_path_test, papers = papers, img_size = img_size)    
    
    #load txt
    txt_path_train = './VQADatasetA_20180815/train.txt'
    txt_path_test  = './VQADatasetA_20180815/test.txt'
    df_txt_train,file_names_train = txt_to_df(txt_path_train)
    df_txt_test ,file_names_test  = txt_to_df(txt_path_test)
    
    #create images input
    X_img_train = image_to_input(img_path_train, file_names_train, papers, img_size, copies)
    X_img_test = image_to_input(img_path_test,file_names_test, papers, img_size, copies)
    
    #create questions and answers input
    df_q_train,df_q_test = question_to_input(list(map(str,df_txt_train['q'])),list(map(str,df_txt_test['q'])))
    df_a_train,ans_list = answer_to_input(df_txt_train)
    
    #data save
    if data_save :
        np.savetxt('X_img_train.txt',X_img_train.reshape(len(X_img_train),-1),fmt = '%.0f',delimiter  = ',')
        np.savetxt('X_img_test.txt' ,X_img_test .reshape(len(X_img_test) ,-1),fmt = '%.0f',delimiter  = ',')
        np.savetxt('df_q_train.txt',df_q_train,fmt = '%.0f',delimiter  = ',')
        np.savetxt('df_q_test.txt' ,df_q_test ,fmt = '%.0f',delimiter  = ',')
        np.savetxt('df_a_train.txt',df_a_train,fmt = '%.0f',delimiter  = ',')
    
#train
    #model
    vqa_model = vqa_model_create(papers, img_size)
    vqa_model.fit([X_img_train,df_q_train], df_a_train, epochs = epochs, batch_size = batch_size)
    
    #model save
    model_name = 'my_model_' + str(img_size) + '_' + str(papers)+ '_' + str(epochs) + '.h5'
    if model_save:
        vqa_model.save(model_name)
        
#predict
    #model load
    if model_load:
        vqa_model = load_model(model_name)
        
    #predict   
    df_a_pre = vqa_model.predict([X_img_test,df_q_test], batch_size= batch_size , verbose=1)
    
    #result
    df_a_pre = np.array(list(map(np.argmax,df_a_pre))).reshape(-1,5)
    result = pd.read_csv(df_txt_test,header=None)
    for index,num in enumerate([2,6,10,14,18]):
    	result[num] = list(map(ans_list.index,df_a_pre[:,index]))
    result.drop([3,4,7,8,11,12,15,16,19,20], axis=1,inplace=True) 
    result.to_csv('pre.txt',header=None,index=None)
    
