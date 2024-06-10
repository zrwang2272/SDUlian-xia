import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses
import argparse
from help_code_demo_tf import ECG_DataSET, ToTensor, create_dataset, FB, Sensitivity  # 您可能需要调整这部分，以确保数据加载和转换与 TensorFlow 兼容
from models.model_tf import AFNet
import numpy as np
import random
import pandas as pd
import os
import time
from cosine_annealing import CosineAnnealingScheduler
from tensorflow.keras import backend as K
from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot


def main():
    seed = 3407
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.config.experimental.enable_op_determinism()
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices
    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []

    # tf.config.experimental.set_visible_devices([], 'GPU')
    # print(tf.config.list_physical_devices('GPU'))
#这里报错，注释掉了
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Instantiating NN
    net = tf.keras.models.load_model('./saved_models/ECG_net_tf_2D_v2.0.h5')
    #net.build(input_shape=(None, 1250, 1,1))
    optimizer = optimizers.Adam(learning_rate=LR)#余弦退火设置
    lrate= CosineAnnealingScheduler(T_max=100, eta_max=4e-4, eta_min=2e-4)
    loss_object = losses.SparseCategoricalCrossentropy(from_logits=True)


    # 定义类别权重
    class_weights = tf.constant([1.0,2.5])
    # 定义加权损失函数
    def weighted_loss(y_true, y_pred):
        # 计算原始损失
        unweighted_loss = loss_object(y_true, y_pred)
        # 计算加权损失
        weights = tf.gather(class_weights, tf.cast(y_true, dtype=tf.int32))
        weighted_loss = tf.math.reduce_mean(weights * unweighted_loss)
        return weighted_loss
        
    # Start dataset loading
    #trainset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='evetu', size=SIZE, transform=ToTensor())
    trainset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices,
                            mode='test', size=SIZE, transform=ToTensor())
    trainloader = create_dataset(trainset, BATCH_SIZE)
    # trainloader = trainloader.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    #testset = ECG_DataSET(root_dir=path_data, indice_dir=path_indices, mode='test', size=SIZE, transform=ToTensor())
    #testloader = create_dataset(testset, BATCH_SIZE)
    # testloader = testloader.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    t=time.strftime('%Y-%m-%d_%H-%M-%S')
    writeFile = open('./saved_models/{}_training_log.txt'.format(t), 'a')
    writeFile.write('4tobe with origin device, no strengthen and 1:2.5 loss\n')
    print("Start training")
    # history = net.fit(trainloader, epochs=EPOCH, validation_data=testloader, verbose=1)
    #step_callback.on_train_begin() # run pruning callback
    for epoch in range(EPOCH):
        #更新学习率
        new_lr=lrate.get_lr(epoch)
        K.set_value(optimizer.lr,new_lr)
        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        for step, (x, y) in enumerate(trainloader):
            #step_callback.on_train_batch_begin(batch=unused_arg) # run pruning callback
            with tf.GradientTape() as tape:
                #print(type(x))
            #correct=m.train(x,y)
                logits = net(x, training=True)
                # loss = loss_object(y, logits)
                loss = weighted_loss(y, logits)
                grads = tape.gradient(loss, net.trainable_variables)
                optimizer.apply_gradients(zip(grads, net.trainable_variables))
                pred = tf.argmax(logits, axis=1)
                correct += tf.reduce_sum(tf.cast(tf.equal(pred, y), tf.float32))
                accuracy += correct / x.shape[0]
                correct = 0.0

                running_loss += loss
                i += 1    
        #step_callback.on_epoch_end(batch=unused_arg) # run pruning callback
        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, accuracy / i, running_loss / i))

        writeFile.write('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, accuracy / i, running_loss / i)+'\n')
        Train_loss.append(running_loss / i)
        Train_acc.append(accuracy / i)

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0
        segs_TP = 0
        segs_TN = 0
        segs_FP = 0
        segs_FN = 0
        subjects_above_threshold = 0

    # Save model
    #net = tfmot.sparsity.keras.strip_pruning(net)
    #net.save('./saved_models/ECG_net_tf_prun.h5')
    net.save('./saved_models/ECG_net_tf_adjustV2.0_Onlytest5+1_lr=0.00015.h5')
    #m.save('/saved_models/modeltest1.ckpt')
    # Write results to file
    file = open('./saved_models/loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')

    print('Finish training')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=1)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.00015)
    argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
    argparser.add_argument('--size', type=int, default=1250)
    argparser.add_argument('--path_data', type=str, default='./Dataset/training_dataset/')
    argparser.add_argument('--path_indices', type=str, default='./data_indices/')
    args = argparser.parse_args()

    main()
