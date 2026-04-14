# CONTROLLING TREE COMPLEXITY AND INTEROPERABILITY
# train the contained model and report training and test accuracy
# display the top five most important features according to the model

# import libraries
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# load dataset, x feature matrix, and y target
data = load_breast_cancer()
x_matrix = data.data
y_target = data.target
names_of_features = data.feature_names

# split data into training and testing sets (80/20 split) with stratification
x_train, x_test, y_train, y_test = train_test_split(x_matrix, y_target, test_size = 0.2, random_state = 42, stratify = y_target)

# using entropy, create a constrained decision tree model
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

# importance scores from decision tree
importance_score_from_decision_tree = decision_tree_model_using_entropy.feature_importances_

# create table to showcase features and their importance scores
features_importance_table = pd.DataFrame({
    "Feature": names_of_features,
    "Importance": importance_score_from_decision_tree
})

# sort features by importance
features_importance_table.sort_values(by = "Importance", ascending = False)

# get the top 5 most important features and print results
top_5_features = features_importance_table.sort_values(by = "Importance", ascending = False).head(5)
print("\nTop 5 Important Features:")
print(top_5_features)

# ------------------------------------------------------------------------------

# EXPLANATIONS
# Controlling model complexity, for example changing the max_depth of the model, allows the model to simplify, thus reducing its complexity
# and constricting the model to rely on memorized data.
# Compared to the uncontrolled model (question 2), the training accuracy for this controlled model was decreased while the testing accuracy was increased
# suggesting a decrease in overfit and improvement with generalization.

# Feature importance provides information regarding the feature that the decision tree model relies on the most to make decisions.
# It shows how much each feature reduces entropy, or uncertainty in the data.
# This allows programmers to learn what feature/features influence outcomes in the decision tree model.