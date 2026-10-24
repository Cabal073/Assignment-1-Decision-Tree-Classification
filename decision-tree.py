from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Split into train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Fit a decision tree on the training set
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Baseline validation accuracy:", model.score(X_val, y_val))

# Tune max_depth using cross-validation
depths = range(1, 11)
depth_scores = []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    scores = cross_val_score(m, X_train, y_train, cv=5, scoring='accuracy')
    depth_scores.append(scores.mean())

# Tune min_samples_leaf using cross-validation
leaves = range(1, 11)
leaf_scores = []
for l in leaves:
    m = DecisionTreeClassifier(min_samples_leaf=l, random_state=42)
    scores = cross_val_score(m, X_train, y_train, cv=5, scoring='accuracy')
    leaf_scores.append(scores.mean())

best_depth = list(depths)[depth_scores.index(max(depth_scores))]
best_leaf = list(leaves)[leaf_scores.index(max(leaf_scores))]
print("Best max_depth:", best_depth)
print("Best min_samples_leaf:", best_leaf)

# Plot hyperparameter values vs performance
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(list(depths), depth_scores, marker='o')
plt.xlabel("max_depth")
plt.ylabel("CV Accuracy")
plt.title("max_depth vs Accuracy")

plt.subplot(1, 2, 2)
plt.plot(list(leaves), leaf_scores, marker='o', color='green')
plt.xlabel("min_samples_leaf")
plt.ylabel("CV Accuracy")
plt.title("min_samples_leaf vs Accuracy")
plt.tight_layout()
plt.show()

# Train on train + validation with optimal hyperparameters
X_trainval = np.vstack([X_train, X_val])
y_trainval = np.hstack([y_train, y_val])
final_model = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=best_leaf, random_state=42)
final_model.fit(X_trainval, y_trainval)
predictions = final_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=data.target_names))

# Visualize the tree
plt.figure(figsize=(20, 10))
plot_tree(final_model, feature_names=data.feature_names, class_names=list(data.target_names), filled=True)
plt.title("Decision Tree")
plt.show()

print("\nTree Rules:")
print(export_text(final_model, feature_names=data.feature_names))