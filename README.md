## Custom implementation of Hyper parameter tunning

1. Implemented the custom RandomSearchCV algorimth to perfoem the hyper-parameter tunning task.
2. Below is te sudo algorithm for RandomSearchCV algorithm.

```python
def RandomSearchCV(x_train, y_train, classifier, param_range, folds):
   x_train: its numpy array of shape, (n,d)
   
   y_train: its numpy array of shape, (n,) or (n,1)
   
   classifier: its typically KNeighborsClassifier()
   
   param_range: its a tuple like (a,b) where a < b
                                           
   folds: an integer, representing the number of folds we need to divide the data and test our model

   1. Generate 10 unique values (uniform random distribution) in the given range "param_range" and store them as "params".
      - Ex: if param_range = (1, 50), generate 10 random numbers between 1 and 50.
   
   2. Divide numbers ranging from 0 to len(X_train) into groups based on the folds.
      - Ex: folds=3, and len(x_train)=100, we can divide numbers from 0 to 100 into 3 groups:
          - Group 1: 0-33
          - Group 2: 34-66
          - Group 3: 67-100

   3. For each hyperparameter generated in step 1:
      - Using the groups created in step 2, perform cross-validation:
        - First, use Group 1+Group 2 (0-66) as train data and Group 3 (67-100) as test data, then calculate train and test accuracies.
        - Second, use Group 1+Group 3 (0-33, 67-100) as train data and Group 2 (34-66) as test data, then calculate train and test accuracies.
        - Third, use Group 2+Group 3 (34-100) as train data and Group 1 (0-33) as test data, then calculate train and test accuracies.
      
      - Repeat this procedure based on the 'folds' value.

      - Find the mean of train accuracies for the above steps and store in a list "train_scores".
      - Find the mean of test accuracies for the above steps and store in a list "test_scores".
   
   4. Return both "train_scores" and "test_scores".

5. Call the function RandomSearchCV(x_train, y_train, classifier, param_range, folds) and store the returned values in "train_score" and "cv_scores".

6. Plot hyper-parameter vs accuracy plot as shown in the reference notebook and choose the best hyperparameter.

7. Plot the decision boundaries for the model initialized with the best hyperparameter, as shown in the last cell of the reference notebook.
