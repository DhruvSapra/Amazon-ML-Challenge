# Amazon-ML-challenge 23'

The goal is to develop a machine learning model that can predict the length dimension of a product. Product length is crucial for packaging and storing products efficiently in the warehouse. Moreover, in many cases, it is an important attribute that customers use to assess the product size before purchasing. However, measuring the length of a product manually can be time-consuming and error-prone, especially for large catalogs with millions of products.

The Training dataset comprises of Product_Title, Description, Bullet_points, Product_ID, Product_Type_ID and Product_Length for 2.2 million products to train and test your submissions.

![download](https://github.com/Dhruv-Sapra/Amazon-ML-challenge/assets/111555972/c4b6b5ac-2eeb-476c-8195-a90be5c9b276)

The working approaches given below follow a machine learning approach that involves several steps, including data preprocessing, feature engineering, model selection, and prediction.

# Approach 1

1. Preprocessing the dataset to remove missing values, fill NaN values with blank strings, remove outliers in numerical data using log1p transformation, and perform min-max scaling. We also carried out text preprocessing by converting all text columns to lower case, dropping punctuations, and lemmatizing the text. This helped us to prepare the dataset for feature engineering.
2. For feature engineering, Implementing one-hot encoding for the categorical column PRODUCT_TYPE_ID, which gave us more features and implemented TF-IDF vectorization on the preprocessed text data to generate additional features.
3. Combining the numerical one-hot and text features using scipy sparse hstack, forming our final training and test features.
4. For our model selection, we experimented with various regression models such as Ridge, SGD and Random forest. We trained each model and evaluated its performance using cross-validation techniques, selecting the best model that gave us the lowest mean squared error (MSE).Ridge Regressor gave us the best result with alpha = 1.0. This model uses L2 regularization to prevent overfitting and produces a linear model that can predict product length accurately.
5. Predicting the results using ridge regression and converting them to the original units by using np.expm1, as we had used log1p to remove outliers during the data preprocessing phase.

# Approach 2

1. Filling any missing or null values in the train and test datasets with an empty string.
2. Filtration of the PRODUCT_LENGTH done for dealing with noise using quantile approach.
3. Removing duplicates based on the 'BULLET_POINTS' column in the train dataset.
4. Creating a new column 'text' in the train and test datasets by copying the 'BULLET_POINTS' column.
5. Converting the text data to lowercase and remove any non-alphanumeric characters from the 'text' column in both the train and test datasets using a lambda function and regular expressions.
6. Creating CountVectorizer and TfidfVectorizer objects to convert the text data into numerical feature vectors. 'stop_words' argument is used to remove commonly occurring words in the English language from the text data.
7. Fitting and transforming the 'text' column of the train dataset using CountVectorizer and TfidfVectorizer, respectively, to obtain the feature vectors of the train dataset.
8. Transforming the 'text' column of the test dataset using the previously fit CountVectorizer and TfidfVectorizer objects to obtain the feature vectors of the test dataset.
9. Concatenating the CountVectorizer and TfidfVectorizer feature vectors of the train and test datasets horizontally using the hstack() method of the scipy.sparse library.
10. Define an instance of the XGBRegressor class which trains model using the fit method.
