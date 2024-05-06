# Constants
IMAGE_SIZE = (32, 32)
NUM_CLASSES = 6
SEED = 42
BATCH_SIZE = 32
EPOCHS = 50



# Constantes modelo 1:
NUM_CONVOLUTIONAL_FILTERS = 4
CONVOLUTIONAL_KERNEL_SIZE = 3
NUM_DENSE_UNITS = 6
IMAGE_CHANNELS = 3
SCALE = 1./255


# Convolutional layer parameters
CONV_FILTERS = [32, 64, 128]
CONV_KERNEL_SIZE = (3, 3)
CONV_ACTIVATION = 'relu'

# Pooling layer parameters
POOL_SIZE = (2, 2)

# Dense layer parameters
DENSE_UNITS = [256, 128]
DENSE_ACTIVATION = 'relu'
DENSE_DROPOUT_RATE = 0.5

# Loss function
LOSS = 'categorical_crossentropy'

# Metrics
METRICS = ['accuracy']
