# MODEL EVALUATION AND COMPARISON
# compute and display the confusion matrix for each model

# import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# load dataset, x feature matrix, and y target
data = load_breast_cancer()
x_matrix = data.data
y_target = data.target

# DECISION TREE
# split data into training and testing sets (80/20 split) with stratification
x_train, x_test, y_train, y_test = train_test_split(x_matrix, y_target, test_size = 0.2, random_state = 42, stratify = y_target)

# using entropy, create decision tree model
decision_tree_model_using_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 42, max_depth = 5)

# train the model
decision_tree_model_using_entropy.fit(x_train, y_train)

# predictions
y_train_prediction = decision_tree_model_using_entropy.predict(x_train)
y_test_prediction = decision_tree_model_using_entropy.predict(x_test)

# compute training and testing accuracy
training_accuracy = accuracy_score(y_train, y_train_prediction)
testing_accuracy = accuracy_score(y_test, y_test_prediction)

# print results
print("The training accuracy is:", training_accuracy)
print("The testing accuracy is:", testing_accuracy)

# NEURAL NETWORK

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

# decision tree confusion matrix
decision_tree_confusion_matrix = confusion_matrix(y_test, decision_tree_model_using_entropy.predict(x_test))

print("Decision Tree Confusion Matrix:")
print(decision_tree_confusion_matrix)

ConfusionMatrixDisplay(decision_tree_confusion_matrix).plot()
plt.title("Decision Tree Confusion Matrix")
plt.show()

# neural network confusion matrix
neural_network_confusion_matrix = confusion_matrix(y_test, neural_network_model.predict(x_test_scaled))

print("\nNeural Network Confusion Matrix:")
print(neural_network_confusion_matrix)

ConfusionMatrixDisplay(neural_network_confusion_matrix).plot()
plt.title("Neural Network Confusion Matrix")
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# EXPLANATIONS
# Model Preference:
# ---> I would prefer the neural network model because it achieves a higher test accuracy
# in comparison to the decision tree model.
# ---> Also is able to capture more complex patterns

# Decision Tree:
# Advantage ---> Easily interpretable decisions
# Limitation ---> Can overfit data if model is not accurately contained

# Neural Network:
# Advantage ---> Ability to capture complex features and provide higher predictions
# Limitation ---> More difficult to interoperate due to more in-depth details