from numpy import transpose, zeros, mean, std, uint8
import os
import scipy.io as sio
from PIL import Image

labels=['normal','inner_1','inner_2','outer_1','outer_2','ball_1','ball_2']

# 12k sampling frequency
fs=12000

#the shape of the every single png
png_shape = (28,28)
sample_len = png_shape[0]*png_shape[1]
#the ratio of dataset
ratio = [0.7, 0.2, 0.1]

#the number of each type
sample_num = 10000

train_num = int(sample_num*ratio[0])
valid_num = int(sample_num*ratio[1])
test_num  = int(sample_num*ratio[2])

#read the ".mat" data
def read_mat(path_file):
    path=[]
    for i in range(len(labels)):
        path.append(os.path.join(os.getcwd(), path_file, labels[i]+'.mat'))
    # read data
    x=zeros((len(labels),fs))
    for i in range(len(labels)):
        # 使用scipy读入mat文件数据
        x_eachType = list(sio.loadmat(path[i]).values())[-2][:fs]
        x[i]=transpose(x_eachType)
    print(x.shape)
    return x

def kurtosis_compute(x, x_mean, x_std):
    K = mean(((x-x_mean)/x_std)**4)
    return K

def create_k_array(x_sample, step):
    k_array = zeros(sample_len)
    k_array2D = zeros(png_shape)
    x_mean = mean(x_sample)
    x_std = std(x_sample)
    for i in range(0,len(x_sample),step):
        k_array[int(i/step)] = kurtosis_compute(x_sample[i:i+step], x_mean, x_std)
    for i in range(png_shape[0]):
        for j in range(png_shape[1]):
            k_array2D[i][j] = k_array[i*png_shape[0]+j]
    return k_array2D
def normal_to_255(x):
    return 255*(x-x.min())/(x.max()-x.min())

def create_png(array):
    array = uint8(array)
    png = Image.fromarray(array)
    return png

def mat_to_png(data_mat):
    train_path=[]
    valid_path=[]
    test_path=[]
    for i in range(len(labels)):
        train_path.append(os.path.join(os.getcwd(), "Data_png", "train", labels[i]))
        valid_path.append(os.path.join(os.getcwd(), "Data_png", "valid", labels[i]))
        test_path.append(os.path.join(os.getcwd(), "Data_png", "test", labels[i]))
        if not os.path.exists(train_path[i]):
            os.makedirs(train_path[i])
        if not os.path.exists(valid_path[i]):
            os.makedirs(valid_path[i])
        if not os.path.exists(test_path[i]):
            os.makedirs(test_path[i])
        
        step=5  #the step for computing kurtosis
        for j in range(0, train_num, 1):
            x_sample = data_mat[i][j:j+sample_len*step]
            k_array2D = create_k_array(x_sample, step)
            png = create_png(normal_to_255(k_array2D))
            save_path = os.path.join(train_path[i],labels[i]+"_"+str(j)+".png")
            png.save(save_path)
            print('\r Process:%.2f%%'%(100*(j+1)/sample_num), end=' ')
        print("\n The training set of the %d th type has done!"%(i+1))
        for j in range(train_num, train_num+valid_num, 1):
            x_sample = data_mat[i][j:j+sample_len*step]
            k_array2D = create_k_array(x_sample, step)
            png = create_png(normal_to_255(k_array2D))
            save_path = os.path.join(valid_path[i],labels[i]+"_"+str(j)+".png")
            png.save(save_path)
            print('\r Process:%.2f%%'%(100*(j+1)/sample_num), end=' ')
        print("\n The validation set of the %d th type has done!"%(i+1))
        for j in range(train_num+valid_num, sample_num, 1):
            x_sample = data_mat[i][j:j+sample_len*step]
            k_array2D = create_k_array(x_sample, step)
            png = create_png(normal_to_255(k_array2D))
            save_path = os.path.join(test_path[i],labels[i]+"_"+str(j)+".png")
            png.save(save_path)
            print('\r Process:%.2f%%'%(100*(j+1)/sample_num), end=' ')
        print("\n The testing set of the %d th type has done!"%(i+1))
    return 0

    
if __name__=='__main__':
    data_mat = read_mat('Data')
    mat_to_png(data_mat)
