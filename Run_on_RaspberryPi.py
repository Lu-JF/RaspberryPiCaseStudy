import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os
from time import sleep

n_class = 7
img_size = 28
BATCH_SIZE = 26
EPOCH = 10

path="./real-data"
labels=['normal','inner_1','inner_2','outer_1','outer_2','ball_1','ball_2']

#reading png and predicting
def read_model_predict(img_path,model):
    img = load_img(img_path,target_size=(img_size, img_size, 3))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    # print(labels[out.argmax()])
    return pred

#the detail of the model out
def detail_show(pred, img_name):
    print('------------------------------------------------')
    for i in range(len(labels)):
        print(img_name+'is %d type with possibility %.6f'%(i+1, pred[i]))

#reading the all files
def read_file_all(model,detail=False):
    path_list=os.listdir(path)
    img_num=0
    for f in path_list:
        image_path = os.path.join(path, f)
        if os.path.isfile(image_path):
            pred = read_model_predict(image_path,model)
            img_num=img_num+1
            name=str(img_num)+'th picture'
        if detail:
            detail_show(pred, name)
        else:
            print(name+'is the type %d with possibility %.4f' % (pred.argmax()+1,pred[pred.argmax()]))

#error handle
def deal_error_command():
    while(True):
        print('Wrong command!')
        command = input('Skip or input again? (S)Skip  (A)input again :')
        if command=='S':
            ifskip=True
            return ifskip
        elif command=='A':
            ifskip=False
            return ifskip
        else:
            continue
#mode config, the default is auto
def ifauto_set():
    ifskip=False
    ifauto=True
    while(not ifskip):
        #接受输入
        command = input('Manual or Automatic? (M)Manual  (A)Automatic :')
        if command=='M':
            ifauto=False
            return ifauto
        elif command=='A':
            ifauto=True
            return ifauto
        else:
            #error handle
            ifskip=deal_error_command()
    return ifauto

#the config of the display
def ifdetail_set():
    ifskip=False
    ifdetail=False
    while(not ifskip):
        command = input('Show detail? (Y)Yse  (N)No :')
        if command=='Y':
            ifdetail=True
            return ifdetail
        elif command=='N':
            ifdetail=False
            return ifdetail
        else:
            #error handle
            ifskip=deal_error_command()
    return ifdetail

def recognize_manual(model,ifdetail):
    ifcontinue=True
    while(ifcontinue):
        command = input('Start to recognize? (Y)Yse  (N)No :')
        if command=='Y':
            read_file_all(model,detail=ifdetail)
            ifcontinue=True
        elif command=='N':
            ifcontinue=False
        else:
            ifskip=deal_error_command()
            #继续就是不跳过
            ifcontinue = not ifskip

    
def recognize_auto(model,ifdetail):
    ifcontinue=True
    while(ifcontinue):
        read_file_all(model,detail=ifdetail)
        sleep(2)


def run():
    print('Start to load the model.')
    model = load_model('model.h5')
    print('The model has finished loading!')
    ifexit=False
    while(not ifexit): 
        #mode config
        ifauto=ifauto_set()
        #display config
        ifdetail=ifdetail_set()
        if ifauto:
            recognize_auto(model,ifdetail)
        else:
            recognize_manual(model,ifdetail)
        command = input('Exit? (Y)Yse  (N)No :')
        if command=='Y':
            ifexit=True
        elif command=='N':
            ifexit=False
        else:
            print('Wrong command!')
            print('Exit directly!')
            ifexit=True



if __name__=='__main__':
    run()
