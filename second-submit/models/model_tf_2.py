import tensorflow as tf
from tensorflow.keras import layers, models


def AFNet():
    model = models.Sequential([
        # layers.Conv2D(filters=17, kernel_size=(10, 1), strides=(6, 1), padding='valid', activation='relu'),

        layers.Conv1D(filters=9, kernel_size=7, strides=2, padding='valid', activation='relu'),
        layers.Conv1D(filters=8, kernel_size=15, strides=2, padding='valid', activation='relu'),
        layers.Conv1D(filters=13, kernel_size=3, strides=2, padding='valid', activation='relu'),
        layers.Conv1D(filters=14, kernel_size=13, strides=2, padding='valid', activation='relu'),

        # layers.Conv1D(filters=8, kernel_size=26, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=7, kernel_size=25, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=10, kernel_size=10, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=20, kernel_size=7, strides=2, padding='valid', activation='relu'),

        # layers.Conv1D(filters=7, kernel_size=25, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=6, kernel_size=25, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=10, kernel_size=10, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=20, kernel_size=6, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=6, kernel_size=27, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=9, kernel_size=34, strides=4, padding='valid', activation='relu'),
        # layers.Conv1D(filters=5, kernel_size=27, strides=2, padding='valid', activation='relu'),

        # layers.Conv2D(filters=9, kernel_size=(7,1), strides=(2,1), padding='valid', activation='relu'),
        # layers.Conv2D(filters=8, kernel_size=(15,1), strides=(2,1), padding='valid', activation='relu'),
        # layers.Conv2D(filters=13, kernel_size=(3,1), strides=(2,1), padding='valid', activation='relu'),
        # layers.Conv2D(filters=14, kernel_size=(13,1), strides=(2,1), padding='valid', activation='relu'),
        
        # layers.Conv1D(filters=5, kernel_size=25, strides=4, padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.Conv1D(filters=8, kernel_size=13, strides=3, padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.Conv1D(filters=6, kernel_size=20, strides=4, padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        
        # layers.Conv1D(filters=11, kernel_size=14, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=8, kernel_size=7, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=15, kernel_size=15, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=7, kernel_size=15, strides=2, padding='valid', activation='relu'),
        # layers.Conv1D(filters=12, kernel_size=14, strides=2, padding='valid', activation='relu'),

        # layers.Conv1D(filters=4, kernel_size=11, strides=2, padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.Conv1D(filters=9, kernel_size=4, strides=2, padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.Conv1D(filters=11, kernel_size=13, strides=2, padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.Conv1D(filters=10, kernel_size=11, strides=2, padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),
        # layers.Conv1D(filters=13, kernel_size=14, strides=2, padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        # layers.Conv2D(filters=5, kernel_size=(5, 1), strides=(2, 1), padding='valid', activation='relu'),
        # # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        # layers.Conv2D(filters=10, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        # layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        # layers.Conv2D(filters=20, kernel_size=(4, 1), strides=(2, 1), padding='valid', activation='relu'),
        # layers.BatchNormalization(epsilon=1e-5, momentum=0.1),

        layers.Flatten(),  # 将卷积层输出扁平化处理，以便输入到全连接层
        # layers.Dropout(0.5),
        # layers.Dense(20, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(2)
    ])

    return model
