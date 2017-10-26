from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from layers import RegionLayer

# region split
region_layer = RegionLayer()
region_layer.split(previous_layer, n_cols=4, n_rows=4)

# add region operations
 region_layer.add(_region_operation)

# region concat
region_concatenated = region_layer.concatenate_convolution()

# region operations
def _region_operation(region):
    region = BatchNormalization()(region)
    region = PReLU()(region)
    region = Convolution2D(16, (3, 3), padding='same',
                           kernel_initializer='he_normal')(region)

    return region