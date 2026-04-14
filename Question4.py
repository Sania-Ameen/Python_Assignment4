# NEURAL NETWORK FOR BINARY CLASSIFICATION
# standardize the input features
# train a neural network with at least one hidden layer and a sigmoid output unit
# report training accuracy and test accuracy

# import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# load dataset, x feature matrix, and y target
data = load_breast_cancer()
x_matrix = data.data
y_target = data.target

# split data into training and testing sets (80/20 split) with stratification
x_train, x_test, y_train, y_test = train_test_split(x_matrix, y_target, test_size = 0.2, random_state = 42, stratify = y_target)

# standardize features to have mean = 0 and ftd = 1
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# create a neural network model
neural_network_model = MLPClassifier(
    hidden_layer_sizes=(10,), # 10 neurons
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

# train the model
neural_network_model.fit(x_train, y_train)

# predictions
y_train_prediction = neural_network_model.predict(x_train)
y_test_prediction = neural_network_model.predict(x_test)

# compute training and testing accuracy
training_accuracy = accuracy_score(y_train, y_train_prediction)
testing_accuracy = accuracy_score(y_test, y_test_prediction)

# print results
print("The training accuracy is:", training_accuracy)
print("The testing accuracy is:", testing_accuracy)

# ------------------------------------------------------------------------------

# EXPLANATION
# Feature scaling is necessary for neural networks because since neural networks are sensitive to input feature scales
# a discrepancy is created and the model becomes biased towards the more larger-scaled features.
# Standardization will allow all features to be introduced more equally.

# An epoch is one pass through an entire training dataset.
# For neural network training, during each epoch, the model would update its weights based on all the
# training samples provided.
# Multiple epochs will allow the model to slowly improve and make fewer prediction errors.

