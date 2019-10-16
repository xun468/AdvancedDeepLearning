import numpy as np
from keras import layers
from keras.layers import *
from keras.models import Model, load_model
from keras.regularizers import l2

def id_block(X, f, filters, stage, block, regularize=True):
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

def conv_block(X, f, filters, stage, block, s = 2):
	#X = Tensor
	#f = shape of conv block,
	#filters=defining the number of filters in the CONV layers 
	#stage = naming thing
	#block = naminh thing
	#s = stride 
	#regularize =  bool determining if we want to use l2 regularization

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    

    F1, F2, F3 = filters
    

    X_shortcut = X


    #First Block
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = 'he_normal')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    #Second Block
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', 
    	kernel_initializer = 'he_normal')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    #third Block
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', 
    	kernel_initializer = 'he_normal')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)


    #shortcut
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = 'he_normal')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X

def resnet(input_shape, n_classes=1000):
	X_input = Input(input_shape)

	X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1, assuming color channel is last, will need to check with real dataset 
	X = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(X)
	X = BatchNormalization(axis=3, name='bn_conv1')(X)
	X = Activation('relu')(X)
	X = MaxPooling2D((3, 3), strides=(2, 2))(X)

	# Stage 2
	X = conv_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
	X = id_block(X, 3, [64, 64, 256], stage=2, block='b')
	X = id_block(X, 3, [64, 64, 256], stage=2, block='c')

	# Stage 3
	X = conv_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
	X = id_block(X, 3, [128, 128, 512], stage=3, block='b')
	X = id_block(X, 3, [128, 128, 512], stage=3, block='c')
	X = id_block(X, 3, [128, 128, 512], stage=3, block='d')

	# Stage 4
	X = conv_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
	X = id_block(X, 3, [256, 256, 1024], stage=4, block='b')
	X = id_block(X, 3, [256, 256, 1024], stage=4, block='c')
	X = id_block(X, 3, [256, 256, 1024], stage=4, block='d')
	X = id_block(X, 3, [256, 256, 1024], stage=4, block='e')
	X = id_block(X, 3, [256, 256, 1024], stage=4, block='f')

	# Stage 5
	X = conv_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
	X = id_block(X, 3, [512, 512, 2048], stage=5, block='b')
	X = id_block(X, 3, [512, 512, 2048], stage=5, block='c')

	#Ending
	X = GlobalAveragePooling2D()(X)
	# X = AveragePooling2D(pool_size=8, name = "pooling", dim_ordering="tf")(X)
	# X = Flatten()(X)
	X = Dense(n_classes, activation='softmax', name='output' + str(n_classes), kernel_initializer = 'he_normal')(X)

	model = Model(inputs = X_input, outputs = X, name='ResNet50')

	return model

# model = resnet((64,64,1),2)