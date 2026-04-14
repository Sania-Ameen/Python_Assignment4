# DATASET EXPLORATION AND UNDERSTANDING
# construct the feature matrix x and target vector y
# report the shape of x and y
# report the number of samples belonging to each class

# import libraries
from sklearn.datasets import load_breast_cancer
import numpy as np

# load the dataset (from the assignment)
data = load_breast_cancer()
print(data)

# create a feature matrix x and target vector y
x_matrix = data.data
y_target = data.target

# check the shape of both x and y
print("\nThe shape of x is:", x_matrix.shape)
print("The shape of y is", y_target.shape)

# count the samples in each class (malignant and benign)
unique, counts = np.unique(y_target, return_counts=True)
class_counts = dict(zip(unique,counts))

# print the number of samples in each class
print("\nThe number of samples per class:")
print("Malignant (0):", class_counts[0])
print("Benign (1):", class_counts[1])

# ------------------------------------------------------------------------------

# EXPLANATIONS
# The given dataset is somewhat imbalanced due to uneven class samples.
# There are more benign (1) samples than malignant (0) samples
# The imbalance in this dataset however, is not extreme.

# Class balance is important due to various reasons, including:
# If one particular class dominates a dataset, than the model may become biased in its prediction.
# In-turn, this may lead to a high accuracy but an overall poor detection of the class in the minority zone.
# In some medical datasets, if there is a discrepancy in class balance, than the chance for misclassification is highly likely and can lead to
# more serious consequences.




