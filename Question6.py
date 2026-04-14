# FASHION MNIST (FOR CNN)
# load the fashion mnist dataset
# normalize the pixel values
# reshape the images gto include the channel dimensions
# train the model for at least 15 epochs
# report the test accuracy

# import libraries
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models

# load dataset
(X_train , y_train ) , (X_test , y_test ) = fashion_mnist.load_data()

# change the data so it is easier to work with (normalize)
X_train = X_train / 255.0
X_test = X_test / 255.0

# reshape by adding channel dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# build the CNN model
cnn_model = models.Sequential ([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# compile the model
cnn_model.compile (optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# train the model
cnn_model.fit(X_train, y_train, epochs = 15) # model will loop.learn 15 times

# evaluate the model
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)

# print test accuracy
print("Test accuracy is:", test_accuracy)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# EVALUATION
# The reason that CNN models are preferred over fully connected networks is because the CNN model
# is able to departure the spatial relationships in images and identifying fine patterns.
# Fully connected networks will treat each individual pixel independently rather than looking at spatial relationships
# in the image. (Not ideal for complex imaged)

# The convolution layer is a part of the CNN model that learns filters which detect image features including
# various different patterns.
# This allows the model to differentiate between the different clothing classes in the given dataset.

