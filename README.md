# Goal

Use various machine learning classification models to predict candidate exoplanet classifications.

# Process

**K-Nearest Neighbors**

After the data was read in from a csv file, some simple cleaning was done by dropping null columns and null rows. After this, there were still several columns available to select as features to train the model on. Wanting to use the most relevant features, I found the top ten features of the data set ranked by feature importances by using `ExtraTreesClassifier()`. I then stored those top ten features as a series to be used as my `X` values in the KNN model. The `koi_disposition` column contained the classification values of each exoplanet candidate and would be used as my `y` values. 

With my `X` and `y` values set, I then split the data into training and testing sets using `train_test_split` with `stratify=y` to ensure that there was an even distribution of classification values in both data sets. Then, I used `MinMaxScaler` to scale both sets of `X` data. 

To find the best K value to use in this model, I created a loop to run through a set of possible K values. 

**Logistic Regression**

**Random Forest**

**Support Vector Machine**
