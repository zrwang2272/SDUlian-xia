import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import argparse
from help_code_demo_tf_2d import ECG_DataSET, ToTensor, create_dataset, FB  # 确保这些数据加载和转换函数与 TensorFlow 兼容
from models.model_tf import AFNet
import numpy as np
import random
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from bayes_opt import BayesianOptimization
from functools import partial
from sklearn.metrics import f1_score, recall_score, fbeta_score
from keras import backend as K
import time

NUM_CLASSES = 10  # Number of classes in the dataset


# 定义初始化模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense


#def get_model(input_shape,fil=3, k_size=20, strs=5,fil_2=5,k_size_2=20,strs_2=5,fil_3=3, k_size_3=20, strs_3=5, fil_4=3, k_size_4=20, strs_4=5):
def get_model(input_shape,dense_1=10,dense_2=2):
    """Builds a Sequential 1D CNN model."""
    # h1=int((1250-int(fil))/strs+1)
    model = Sequential([
        Conv2D(filters=3, kernel_size=(15,1), strides=(2,1), padding='valid', activation='relu', name="conv2d_1"),
        Conv2D(filters=11, kernel_size=(15,1), strides=(2,1), padding='valid', activation='relu',  name="conv2d_2"),
        Conv2D(filters=12, kernel_size=(9,1), strides=(2,1), padding='valid', activation='relu',  name="conv2d_3"),
        Conv2D(filters=15, kernel_size=(20,1), strides=(3,1), padding='valid', activation='relu',  name="conv2d_4"),
        # layers.MaxPooling2D(pool_size=(2,1)),
        Flatten(name="flatten"),
        Dense(int(dense_1), activation='relu', name="dense_1"),
        Dense(int(dense_2), activation=None, name="dense_2")
    ])
    return model

#设置随机数
#seed = 222
seed = 3407
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
tf.config.experimental.enable_op_determinism()
# 设置GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# 数据集加载路径
path_data = './Dataset/training_dataset/'
path_indices = './data_indices/'
SIZE = 1250
BATCH_SIZE = 32
# 数据加载
trainset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='train', size=SIZE, transform=ToTensor())
trainloader = create_dataset(trainset, BATCH_SIZE)
testset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE, transform=ToTensor())
testloader = create_dataset(testset, BATCH_SIZE)

maxfbeta=0.95

def fit_with(d1,d2, verbose, lr):
    input_shape = (1250, 1, 1)
    # 使用指定的超参数创建模型
    model = get_model(input_shape, int(d1),int(d2))
    
    print("Start training")

    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(trainloader, epochs=5, verbose=2)  # 每一个训练15个epoch
    test_indice_path = args.path_indices + 'test_indice.csv'
    test_indices = pd.read_csv(test_indice_path)  # Adjust delimiter if necessary
    subjects = test_indices['Filename'].apply(lambda x: x.split('-')[0]).unique().tolist()
    subjects_above_threshold = 0
    avg_fb = 0 
    for subject_id in subjects:
        segs_2TP = 0
        segs_2TN = 0
        segs_2FP = 0
        segs_2FN = 0
        test2set = ECG_DataSET(root_dir=path_data,
                               indice_dir=path_indices,
                               mode='test',
                               size=SIZE,
                               subject_id=subject_id,
                               transform=ToTensor())

        test2loader = create_dataset(test2set, 1)
        for ECG_test, labels_test in test2loader:
            predictions = model(ECG_test, training=False)
            predicted_test = tf.argmax(predictions, axis=1)

            seg_2label = labels_test.numpy()[0]
            
            if seg_2label == 0:
                segs_2FP += np.sum(predicted_test.numpy() != labels_test.numpy())
                segs_2TN += np.sum(predicted_test.numpy() == labels_test.numpy())
            elif seg_2label == 1:
                segs_2FN += np.sum(predicted_test.numpy() != labels_test.numpy())
                segs_2TP += np.sum(predicted_test.numpy() == labels_test.numpy())
        fb = round(FB([segs_2TP, segs_2FN, segs_2FP, segs_2TN]), 5)
        avg_fb += fb
        print(f"{subject_id}:{fb:.5f}")
        if fb > 0.9:
            subjects_above_threshold += 1
                
    proportion_above_threshold = subjects_above_threshold / len(subjects)
    print("G Score:", proportion_above_threshold)
    avg_fb /= len(subjects)
    print(f"Final avg F-B: {avg_fb:.5f}")
    print("d1:", d1)
    print("d2:", d2)
    if(proportion_above_threshold>0.9 and maxfbeta<avg_fb):
        maxfbeta=avg_fb
        model.save('./saved_models/g9tobe.h5')
    return avg_fb  # 以F-B作为其指标

def main():
    

    t=time.strftime('%Y-%m-%d_%H-%M-%S')
    writeFile = open('./saved_models/{}_training_log.txt'.format(t), 'a')
    writeFile.write('new new new 2D 4 tobe with original device\n')

    # Hyperparameters
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices



    verbose = 2
    fit_with_partial = partial(fit_with, verbose=verbose, lr=LR)
    
    # 定义参数空间
    # pbounds = {'fil':(3,7),'k_size': (5, 35), 'strs': (2, 5),'fil_2':(5,10),'k_size_2': (5,35), 'strs_2': (2,5),
    #             'fil_3':(5,10),'k_size_3': (5,35), 'strs_3': (2,5),'fil_4':(5,10),'k_size_4': (5,35), 'strs_4': (2,5)}
    pbounds = {'d1': (7,15), 'd2':(2,7)}
    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    #搜索次数
    optimizer.maximize(init_points=10, n_iter=10)

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}\n\t".format(i, res))
        writeFile.write("Iteration {}: \n\t{}\n\t".format(i, res))

    print(optimizer.max)
    writeFile.write(str(optimizer.max))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=7)#30次搜
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.001)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./Dataset/training_dataset/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    args = argparser.parse_args()

    main()

