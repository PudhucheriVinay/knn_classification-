# K-Nearest Neighbors (KNN) Classification

## Objective
This project demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm for classification problems using the Iris dataset. The goal is to understand and apply KNN, experiment with different values of K, evaluate the model, and visualize decision boundaries.

## Tools Used
- Python 3
- Scikit-learn
- Pandas (not used explicitly but can be added for dataset handling)
- Matplotlib

## Project Structure
- `knn_classification.py`: Main script that loads the Iris dataset, normalizes features, trains KNN classifiers with different K values, evaluates accuracy and confusion matrix, and visualizes decision boundaries.

## How to Run
1. Ensure you have Python 3 installed.
2. Install required packages:

   pip install numpy matplotlib scikit-learn

3. Run the script:
  
   python knn_classification.py

4. The script will print accuracy and confusion matrix for different K values and display plots of decision boundaries.

## Interview Questions and Answers

1. **How does the KNN algorithm work?**  
   KNN is an instance-based learning algorithm that classifies a data point based on the majority class among its K nearest neighbors in the feature space, typically using Euclidean distance.

2. **How do you choose the right K?**  
   The right K is chosen by experimenting with different values and selecting the one that provides the best performance on validation data. Too small K can be noisy, too large K can smooth out class boundaries.

3. **Why is normalization important in KNN?**  
   Normalization ensures that all features contribute equally to the distance calculation. Without normalization, features with larger scales can dominate the distance metric.

4. **What is the time complexity of KNN?**  
   The time complexity is O(n) for each prediction, where n is the number of training samples, since it computes distances to all training points.

5. **What are pros and cons of KNN?**  
   Pros: Simple, effective for small datasets, no training phase.  
   Cons: Slow prediction for large datasets, sensitive to irrelevant features and noise.

6. **Is KNN sensitive to noise?**  
   Yes, KNN can be sensitive to noisy data points because it relies on local neighbors for classification.

7. **How does KNN handle multi-class problems?**  
   KNN naturally handles multi-class classification by voting among neighbors belonging to multiple classes.

8. **What’s the role of distance metrics in KNN?**  
   Distance metrics determine how similarity between points is measured, affecting neighbor selection and classification accuracy.


