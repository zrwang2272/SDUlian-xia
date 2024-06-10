# custom.py
# 注册CONV1D_SDU  

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import ops 

# Assuming `conv1d_sdu_op.so` is the compiled shared library for the custom op
conv1d_sdu_module = tf.load_op_library('conv1d_sdu_op.so')

@ops.RegisterGradient("Conv1dSdu")
def _conv1d_sdu_grad(op, grad):
    return [conv1d_sdu_module.conv1d_sdu_grad(op.inputs[0], op.inputs[1], grad)]

class Conv1D_SDU(Layer):
    def __init__(self, kernel, dilation, **kwargs):
        super(Conv1D_SDU, self).__init__(**kwargs)
        self.kernel = kernel
        self.dilation = dilation

    def build(self, input_shape):
        super(Conv1D_SDU, self).build(input_shape)

    def call(self, inputs):
        return conv1d_sdu_module.conv1d_sdu(inputs, kernel=self.kernel, dilation=self.dilation)

    def compute_output_shape(self, input_shape):
        return input_shape  # Adjust this according to your op's logic