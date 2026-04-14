# DECISION TREE MODEL USING ENTROPY
# train a decision using entropy
# report the training accuracy and test accuracy of model

# import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load dataset, x feature matrix, and y target
data = load_breast_cancer()
x_matrix = data.data
y_target = data.target

# split data into training and testing sets (80/20 split) with stratification
x_train, x_test, y_train, y_test = train_test_split(x_matrix, y_target, test_size = 0.2, random_state = 42, stratify = y_target)

# using entropy, create decision tree model
decision_tree_model_using_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 42)

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

# ------------------------------------------------------------------------------

# EXPLANATIONS
# Entropy measures the uncertainty or randomness that persists in the data.
# In a decision tree, entropy will tell programmers or users how mixed classes are in a singular node.
# Using information gain, the model will choose a split that will allow a reduction in entropy thus providing lower uncertainty.

# The results from this data suggest that the model is overfitting.
# This can be concluded based off of the training accuracy and the test accuracy scores -
# the training accuracy is much higher, meaning that the model is mostly memorizing training data.
# However, since both training and testing accuracies are high, we can conclude an overall good generalization.