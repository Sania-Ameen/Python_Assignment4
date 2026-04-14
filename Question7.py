# CNN ERROR ANALYSIS AND MISCLASSIFICATION STUDY
# generate predictions on the test set
# compute and display the confusion matrix
# identify and visualize at least three misclassified images
# show the true label and predicted label

# import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

# get predicted probabilities from CNN model and convert them
y_predicted_probabilities = cnn_model.predict(X_test)

y_predicted = np.argmax(y_predicted_probabilities, axis = 1)

# construct, print and display the confusion matrix
confusion_matrix_of_cnn = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:")
print(confusion_matrix_of_cnn)

ConfusionMatrixDisplay(confusion_matrix_of_cnn).plot()
plt.title("CNN Confusion Matrix")
plt.show()

# get misclassified images
# get places where the prediction is wrong and take the first 3 incidents
misclassified_indices = np.where(y_predicted != y_test)[0]
sample_indices = misclassified_indices[:3]
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# show misclassified images
plt.figure(figsize=(10, 4))

for i, idx in enumerate(sample_indices):
    plt.subplot(1, 3, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')

    plt.title(
        f"True: {class_names[y_test[idx]]}\n"
        f"Pred: {class_names[y_predicted[idx]]}"
    )

    plt.axis('off')

plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# EVALUATION
# One pattern that I observed in the misclassifications, is that the model often confuses
# similar looking clothing items with one another since they have similar shapes/patterns.

# One realistic method to improve the CNN performance is to add more convolutional layers
# so that the model can learn to pick up the more fine details in the images.


