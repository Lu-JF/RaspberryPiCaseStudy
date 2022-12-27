# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 01:28:34 2022

@author: LJF
"""
from keras.preprocessing.image import ImageDataGenerator
from Nets import googlenet, RMFF_CNN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools

labels=['normal','inner_1','inner_2','outer_1','outer_2','ball_1','ball_2']

#the shape of the every single png
png_shape = (28,28)
#the ratio of dataset
ratio = [0.7, 0.2, 0.1]
#the number of each type
sample_num = 10000
train_num = int(sample_num*ratio[0])
valid_num = int(sample_num*ratio[1])
test_num  = int(sample_num*ratio[2])
n_class = 7
img_size = 28
BATCH_SIZE = 32
EPOCH = 5

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
	#plt.savefig('fusion_matrix.png',dpi=350)
    plt.show()

def plot_fusion(model, data_generator):
    data = data_generator.next()
    predictions = model.predict(data[0])
    predictions = predictions.argmax(axis=-1)
    label = data[1].argmax(axis=-1)
    conf_mat = confusion_matrix(y_true=label, y_pred=predictions)
    plot_confusion_matrix(conf_mat, normalize=False,target_names=labels,title='Confusion Matrix')


if __name__=='__main__':
    #data loading
    train_datagen = ImageDataGenerator()
    valid_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow_from_directory(
            './/Data_png//train',
            target_size=png_shape,
            batch_size=BATCH_SIZE,
            class_mode='categorical')
    validation_generator = valid_datagen.flow_from_directory(
            './/Data_png//valid',
            target_size=png_shape,
            batch_size=BATCH_SIZE,
            class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(
            './/Data_png//test',
            target_size=png_shape,
            batch_size=test_num,
            class_mode='categorical')
    #creating and training the model
    # model = googlenet(n_class)
    model = RMFF_CNN(n_class)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    H=model.fit(
        train_generator,
        steps_per_epoch=train_generator.n/BATCH_SIZE,
        epochs=EPOCH,
        validation_data=validation_generator,
        validation_steps=validation_generator.n/BATCH_SIZE)
    model.save("model.h5")
    # plot the training process
    train_H = H.history["accuracy"]
    val_H = H.history["val_accuracy"]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCH), train_H, linestyle='-', label="train_acc")
    plt.plot(np.arange(0, EPOCH), val_H, linestyle='dotted', label="valid_acc")
    plt.xlabel('Trial Number')
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    # plt.savefig("plot.png")
    plt.show()
    #model test and plot the confusion matrix
    plot_fusion(model, test_generator)