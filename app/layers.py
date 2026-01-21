import tensorflow as tf
from tensorflow.keras.layers import Layer
## this is done beacuse the this is a custom funvtion and required every time we load the custom model
class L1dist(Layer):   ## custom l1 distance function ( layer )
    def __init__(self,**kwargs):
        super().__init__()
    def call(self, input_embedding,val_embedding):
        return tf.math.abs(input_embedding-val_embedding)
    def get_config(self):
        return super().get_config()