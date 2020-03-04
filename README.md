# Goal

Use various machine learning classification models to predict candidate exoplanet classifications. Use Grid Search to increase the accuracy of the model. 

# Process

**Data Cleaning and Pre-Processing**

Data was first read in from a csv file, and null columns and null rolls were dropped. After this, there were still several columns available to select as features to train the model on. Wanting to use the most relevant features, I found the top ten features of the data set ranked by feature importances by using `ExtraTreesClassifier()` and stored those top ten features as a series to be used as my `X` values. The `koi_disposition` column contained the classification values of each exoplanet candidate and would be used as my `y` values. 

With my `X` and `y` values set, I split the data into training and testing sets using `train_test_split` with `stratify=y` to ensure that there was an even distribution of classification values in both data sets. Then, I used `MinMaxScaler` to scale both sets of `X` data.

This method was used for all four models.

**K-Nearest Neighbors**

To find the best k value to use in this model, I created a loop to run through a set of possible k values. Because there are three possible classifications, I started the range of k values at 5 with a step of 3 to avoid any even split of classifications. Comparing the training and testing scores of each model, it looked like k=14 was the best value, as it had the lowest difference between training and testing scores, without the testing score being higher than the training (which suggests overfitting).

To further tune the model’s parameters, I used `GridSearchCV` and expanded the possible values of k. I then retrained the model using the best parameter found and scored the model using the test set of data. 

Grid Search also found k=14 to be the best k value, with an accuracy of 88.5%, so this model was not improved by the use of Grid Search.

**Logistic Regression**

**Random Forest**

**Support Vector Machine**
