import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy import stats

lemmatizer = WordNetLemmatizer()

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


train_df.head()
test_df.head()


train_df.info()
test_df.info()


# Remove duplicate rows if any
train_df = train_df.drop_duplicates()
test_df = test_df.drop_duplicates()
print("dropped Dups")

#Fill missing values with appropriate methods:
train_df.isnull().sum()
test_df.isnull().sum()

train_df["TITLE"].fillna("", inplace=True)
train_df["DESCRIPTION"].fillna("", inplace=True)
train_df["BULLET_POINTS"].fillna("", inplace=True)

test_df["TITLE"].fillna("", inplace=True)
test_df["DESCRIPTION"].fillna("", inplace=True)
test_df["BULLET_POINTS"].fillna("", inplace=True)


#For the PRODUCT_TYPE_ID column, you can fill missing values with the mode of the column:
train_df["PRODUCT_TYPE_ID"].fillna(train_df["PRODUCT_TYPE_ID"].mode()[0], inplace=True)
test_df["PRODUCT_TYPE_ID"].fillna(train_df["PRODUCT_TYPE_ID"].mode()[0], inplace=True)


#For the PRODUCT_LENGTH column, you can drop the rows with missing values since it's the target variable:
train_df.dropna(subset=["PRODUCT_LENGTH"], inplace=True)

print("filled missing")

# #Z-score step
# # Identify and remove outliers using the Z-score method
# z_scores = stats.zscore(train_df[["PRODUCT_LENGTH"]])
# abs_z_scores = np.abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# train_df = train_df[filtered_entries]

# # Verify and clean incorrect data values
# train_df["PRODUCT_TYPE_ID"] = train_df["PRODUCT_TYPE_ID"].astype(int)
# train_df["PRODUCT_LENGTH"] = train_df["PRODUCT_LENGTH"].astype(float)

# # Normalize the data by scaling the values
# train_df["PRODUCT_LENGTH"] = (train_df["PRODUCT_LENGTH"] - train_df["PRODUCT_LENGTH"].min()) / (train_df["PRODUCT_LENGTH"].max() - train_df["PRODUCT_LENGTH"].min())
# print("Z step")

#Concatenate the TITLE, DESCRIPTION, and BULLET_POINTS columns into a single column TEXT:  
train_df["TEXT"] = train_df["TITLE"] + " " + train_df["DESCRIPTION"] + " " + train_df["BULLET_POINTS"]
test_df["TEXT"] = test_df["TITLE"] + " " + test_df["DESCRIPTION"] + " " + test_df["BULLET_POINTS"]

train_df.drop(columns=["TITLE", "DESCRIPTION", "BULLET_POINTS"], inplace=True)
test_df.drop(columns=["TITLE", "DESCRIPTION", "BULLET_POINTS"], inplace=True)
print("Concate done")
#Remove special characters and punctuation marks

train_df["TEXT"] = train_df["TEXT"].apply(lambda x: re.sub(r'[^\w\s]', '', x))
test_df["TEXT"] = test_df["TEXT"].apply(lambda x: re.sub(r'[^\w\s]', '', x))
print("special char removed")
#Remove stop words: and the 

stop_words = set(stopwords.words('english'))

train_df["TEXT"] = train_df["TEXT"].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
test_df["TEXT"] = test_df["TEXT"].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
print("stop words removed")
#Lemmatize the text running to run

train_df["TEXT"] = train_df["TEXT"].apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split()))
test_df["TEXT"] = test_df["TEXT"].apply(lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split()))
print("lemmatized")

#Convert text to lowercase
train_df["TEXT"] = train_df["TEXT"].apply(lambda x: x.lower())
test_df["TEXT"] = test_df["TEXT"].apply(lambda x: x.lower())
print("lower cased")

# Save the cleaned dataset to a new CSV file
train_df.to_csv('cleaned_train.csv', index=False)
test_df.to_csv("cleaned_test.csv", index=False)


