import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import argparse
from help_code_demo_tf import ECG_DataSET, ToTensor, create_dataset, FB  # 确保这些数据加载和转换函数与 TensorFlow 兼容
from models.model_tf import AFNet
import numpy as np
import random
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPooling1D, Flatten
from bayes_opt import BayesianOptimization
from functools import partial
from sklearn.metrics import f1_score, recall_score, fbeta_score
from keras import backend as K
import time

NUM_CLASSES = 10  # Number of classes in the dataset


# 定义初始化模型
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense


def get_model(input_shape,fil=3, k_size=20, strs=3,fil_2=5,k_size_2=20,strs_2=3,fil_3=3, k_size_3=20, strs_3=3,fil_4=3, k_size_4=20, strs_4=3):
    """Builds a Sequential 1D CNN model."""
    # h1=int((1250-int(fil))/strs+1)
    model = Sequential([
        Conv1D(filters=int(fil), kernel_size=int(k_size), strides=2, padding='valid', activation='relu', name="conv1d_1"),
        Conv1D(filters=int(fil_2), kernel_size=int(k_size_2), strides=int(strs_2), padding='valid', activation='relu',  name="conv1d_2"),
        Conv1D(filters=int(fil_3), kernel_size=int(k_size_3), strides=int(strs_3), padding='valid', activation='relu',  name="conv1d_3"),
        Conv1D(filters=int(fil_4), kernel_size=int(k_size_4), strides=int(strs_4), padding='valid', activation='relu',  name="conv1d_4"),
        Flatten(name="flatten"),
        # Dense(20, activation='relu', name="dense_1"),
        Dense(10, activation='relu', name="dense_2"),
        Dense(2, activation=None, name="dense_3")
    ])
    return model

def fit_with(fil,k_size, strs,fil_2,k_size_2,strs_2, fil_3, k_size_3, strs_3,fil_4, k_size_4, strs_4,verbose, lr):
    def fbeta(y_true, y_pred, beta=1):
        # 计算精确率和召回率
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
        fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
        fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)
        
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        
        # 计算 F-beta 分数
        beta_squared = beta ** 2
        fbeta_score = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
        
        return K.mean(fbeta_score)
    input_shape = (1250, 1)
    # 使用指定的超参数创建模型
    model = get_model(input_shape,int(fil), int(k_size), int(strs),int(fil_2),int(k_size_2),int(strs_2),
                      int(fil_3), int(k_size_3), int(strs_3),int(fil_4), int(k_size_4), int(strs_4))
    
    # 数据集加载路径
    path_data = args.path_data
    path_indices = args.path_indices
    SIZE = args.size
    BATCH_SIZE = args.batchsz

    # 数据加载
    trainset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='train', size=SIZE, transform=ToTensor())
    trainloader = create_dataset(trainset, BATCH_SIZE)

    testset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE, transform=ToTensor())
    testloader = create_dataset(testset, BATCH_SIZE)

    print("Start training")

    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(trainloader, epochs=5, verbose=2)  # 每一个训练15个epoch
    score = model.evaluate(testloader, verbose=2)

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
        # writeFile.write(f"{subject_id}:{fb:.5f}\n")
        if fb > 0.9:
            subjects_above_threshold += 1
                
    proportion_above_threshold = subjects_above_threshold / len(subjects)
    print("G Score:", proportion_above_threshold)
    avg_fb /= len(subjects)
    print(f"Final avg F-B: {avg_fb:.5f}")
    return avg_fb  # 以ACC作为其指标

def main():
    seed = 3407
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()

    t=time.strftime('%Y-%m-%d_%H-%M-%S')
    # writeFile = open('./saved_models/{}_training_log.txt'.format(t), 'a')
    # writeFile.write('3 tobe with original device and strengthen\n')

    # Hyperparameters
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    # 设置GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    verbose = 2
    fit_with_partial = partial(fit_with, verbose=verbose, lr=LR)
    
    # 定义参数空间
    pbounds = {'fil':(3,7),'k_size': (5,20), 'strs': (2, 3),'fil_2':(7,12),'k_size_2': (5,20), 'strs_2': (2,4),
               'fil_3':(10,16),'k_size_3': (5,20), 'strs_3': (2,4),'fil_4':(10,16),'k_size_4': (5,18), 'strs_4': (2,4)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    optimizer.maximize(init_points=10, n_iter=25)

    # for i, res in enumerate(optimizer.res):
    #     print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)#30次搜
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.0005)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./Dataset/training_dataset/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    args = argparser.parse_args()

    main()
