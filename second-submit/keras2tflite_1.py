import tensorflow as tf

# Create a model using high-level tf.keras.* APIs
model = tf.keras.models.load_model('./saved_models/ECG_net_tf.h5')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#允许自己写算子
converter.allow_custom_ops = True

tflite_model = converter.convert()

# Save the model.
with open('saved_models/model.tflite', 'wb') as f:
  f.write(tflite_model)
