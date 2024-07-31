import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Is GPU available?", tf.config.list_physical_devices('GPU'))
print("Is TensorFlow built with CUDA?:", tf.test.is_built_with_cuda())
# yay it works on my computer wooohoo
