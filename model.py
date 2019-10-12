import numpy as np
from keras import layers
from keras.layers import *
from keras.models import Model, load_model

def identity_block(X, f, filters, stage, block, regularize=True):
	#X = Tensor
	#f = shape of conv block,
	#filters=defining the number of filters in the CONV layers 
	#stage = naming thing
	#block = naminh thing
	#regularize =  bool determining if we want to use l2 regularization
    #he initalization chosen in line with keras implementation 

    conv_name_base = 'identity_conv' + str(stage) + block + '_'
    bn_name_base = 'bn' + str(stage) + block + '_'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    # First block
    if(regularize):
    	X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = 'he_normal',
    		kernel_regularizer=l2(1e-4))(X)
    else:
    	X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = 'he_normal')(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    
    # Second block
    if(regularize):
    	X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = 'he_normal',
    		kernel_regularizer=l2(1e-4))(X)
    else:
    	X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)

    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third block
    if(regularize):
    	X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = 'he_normal',
    		kernel_regularizer=l2(1e-4))(X)
    else:
    	X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final block
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X

def convolutional_block():
	return

def resnet(input_shape, n_classes=1000):
	X_input = Input(input_shape)

	X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1, assuming color channel is last, will need to check with real dataset 
    X = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    #Stage 2
	#convblock
	#id
	#id 
	model = models.Model(inputs, x, name='resnet50')
	return model