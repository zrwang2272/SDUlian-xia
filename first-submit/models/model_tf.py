import tensorflow as tf
from tensorflow.keras import layers, models

def AFNet():
    model = models.Sequential([
        # bestnow
        layers.Conv2D(filters=3, kernel_size=(15,1), strides=(2,1), padding='valid', activation='relu'),
        layers.Conv2D(filters=11, kernel_size=(15,1), strides=(2,1), padding='valid', activation='relu'),
        layers.Conv2D(filters=12, kernel_size=(9,1), strides=(2,1), padding='valid', activation='relu'),
        layers.Conv2D(filters=15, kernel_size=(20,1), strides=(3,1), padding='valid', activation='relu'),
        layers.Flatten(),  # 将卷积层输出扁平化处理，以便输入到全连接层
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model
