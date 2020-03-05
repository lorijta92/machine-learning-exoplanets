# Goal

Use various machine learning classification models to predict candidate exoplanet classifications. Use Grid Search to increase the accuracy of the model. 

# Process

**Data Cleaning and Pre-Processing**

Data was first read in from a csv file, and null columns and null rolls were dropped. After this, there were still several columns available to select as features to train the model on. Wanting to use the most relevant features, I found the top ten features of the data set ranked by feature importances by using `ExtraTreesClassifier()` and stored those top ten features as a series to be used as my `X` values. The `koi_disposition` column contained the classification values of each exoplanet candidate and would be used as my `y` values. 

With my `X` and `y` values set, I split the data into training and testing sets using `train_test_split` with `stratify=y` to ensure that there was an even distribution of classification values in both data sets. Then, I used `MinMaxScaler` to scale both sets of `X` data.

This method was used for all four models.

**K-Nearest Neighbors**

To find the best k value to use in this model, I created a loop to run through a set of possible k values. Because there are three possible classifications, I started the range of k values at 5 with a step of 3 to avoid any even split of classifications. Comparing the training and testing scores of each model, it looked like k=17 was the best value, as it had the lowest difference between training and testing scores, without the testing score being higher than the training.

To further tune the model’s parameters, I used `GridSearchCV` and expanded the possible values of k. I then retrained the model using the best parameter found and scored the model using the test set of data.

Grid Search also found k=48 to be the best k value, with an accuracy of 86%, so this model was not improved by the use of Grid Search.

**Logistic Regression**

I initialized the model using `LogisticRegression()` and fit the model using the training data. I then scored the model using both the training and testing data. Both sets scored fairly well, with the training data at 84% and the testing data at 84.3%.

I again used `GridSearchCV` to further tune the parameters to create a better scoring model. The parameters were set to explore different `C` values using both L1 and L2 penalties as regularization methods. I then fit a new model using this grid and found the best parameters, before predicting on the test data. This new model's score was better than the original by 3.4%, scoring at 87.7%.

**Random Forest**

I initialized the model using `RandomForestClassifier()` and set the number of trees to 300 (`n_estimators=300`). I then fit and scored the model, with the testing data scoring at 88.7%.

Using Grid Search, I explored different parameters including `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`. Again, I used this grid to train a new model, before predicting and scoring. The new model scored at 89.1%, only slightly improved from the original model.

**Support Vector Machine**

I initialized the model with `SVC()` and set the kernel to `linear` before training and scoring the model, with the testing data scoring at 80.6%.

With Grid Search, I explored various C values, gamma values, and linear and rbf kernels. After training the new model, accuracy increased to 84.2%.
