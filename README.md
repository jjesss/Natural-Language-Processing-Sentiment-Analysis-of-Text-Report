# Natural-Language-Processing-Sentiment-Analysis-of-Text-Report

## Introduction
This mini research project into Machine Learning is focused on solving the problem of automatically
classifying recorded tweets from Twitter into sentimental categories. This is important since with an
increase in technology and social media, more and more people are being accustomed to expressing
their opinions online on social media platforms such as Twitter. Therefore, through this sentimental
analysis, we can have an overview of people’s general feelings and opinions on a variety of specific
topics. This can be useful in helping companies and businesses to gather feedback of their customers
and monitor the brand’s reputation much more easily and quickly.
In this project I am aiming to train a dataset in three different algorithm models and test to see the
accuracy of each prediction. I also aim to use the confusion matrix to gain more insight on the
percentage of wrong and right predictions of each model. In gaining the accuracy of each model’s
predictions and the percentage of wrong and rights, I aim to identify the model that would be the
most reasonable to use with my problem of automatically classifying recorded tweets from Twitter
into sentimental categories.

## Data and Preperation
The dataset I used to investigate this problem is in a csv file containing two parts, a ‘content’ part
which contains the text of the tweet and a corresponding ‘sentiment’ part which determines which
of the 4 sentiments the corresponding tweet is assigned to/ describes the sentiment of the tweet.
The 4 sentiment categories are relief, happiness, surprise, and enthusiasm. This dataset will need to
be split into a training section of training features and labels which will train the algorithm models
and then these models will be tested using the remaining data, the testing section. The model uses
the testing features to make predictions which are then compared with their labels.
After importing the csv file into MATLAB as a table, I built a Bag of Words containing all the
tokenised tweets, removed top words and removed any words with fewer than 100 occurrences in
the bag. Then I built the Term Frequency-Inverse Document Frequency matrix for the resulting bag
as well as the corresponding label vector from the column of sentiments. The dataset was then split
into a section for training and one for testing. I used the first 6432 rows of the tf-idf matrix for the
training features and the corresponding sentiments for the training labels and the rest of the rows in
the tf-idf matrix and sentiments for testing features and testing labels respectively.

## Metholodogy
The training data that was made in the preparation steps before is used to train three classification
algorithms with the training features and training labels being the parameters used in the functions
implemented in MATLAB.
The classification algorithms I used was K-Nearest Neighbour, Decision Tree and Discriminant
Analysis. K-Nearest Neighbour uses proximity to make classifications about the grouping of an
individual data, selecting the K number of points which is closest to the test data and chooses the
class of the majority of those points. The Decision Tree model uses a hierarchical tree structure. To
choose the class for the data, the data is continuously split according to a certain parameter until it
reaches a leaf node which corresponds to a class label. Then the Discriminant Analysis model
estimates the probability of data belonging to every class and chooses the highest probability. I
chose these models since both decision trees and discriminant analysis have fast prediction speed as
well as easy interpretability. As these both have fast prediction speeds, I chose K-nearest neighbour
to compare which has medium prediction speed. Despite this it has no assumptions about
underlying data so it’s good for non-linear data so I wanted to see what the accuracy level would be
like compared to the other two quicker models.
After the models was trained, the testing data was used to evaluate the accuracy of each model. The
trained model was used to predict the labels of the testing features. Then the accuracy was
calculated by how many labels were predicted correctly out of the total amount of testing labels.

## Results
K-Nearest Neighbour model has a prediction accuracy of 36%. The Discriminant Analysis model has a
prediction accuracy of 50% and the Decision Tree has one of 43%. Meaning the overall the
Discriminant Analysis model is the most accurate at predicting the sentimental labels of the Tweets
in the dataset.
The confusion matrices for the three models:

![image](https://github.com/jjesss/Natural-Language-Processing-Sentiment-Analysis-of-Text-Report/assets/77901330/6c6db337-9e60-41b0-aab5-e4cc962813a4)

From the confusion table, the following measures are calculated: (to the nearest percentage)

|                      | K-Nearest Neighbour | Discriminant Analysis | Decision Tree |
|----------------------|--------------------|----------------------|---------------|
| Precision of enthusiasm | 11%                | 27%                  | 5%            |
| Precision of happiness  | 55%                | 51%                  | 50%           |
| Precision of relief     | 21%                | 39%                  | 57%           |
| Precision of surprise   | 20%                | 24%                  | 21%           |
| Overall accuracy        | 36%                | 50%                  | 43%           |

From the table we see the most precise model for predicting enthusiasm is the Discriminant Analysis
model, for happiness it’s the K-Nearest neighbour, for relief it’s the Decision Tree and for surprise it’s
again the Discriminant Analysis.

## Conclusion 
In conclusion, I would recommend using the Discriminant Analysis model for solving this problem of
automatically classifying recorded tweets from Twitter into sentimental categories. This is because
overall it has the highest accuracy as well as the best prediction precision for two of the four
sentimental categories. Not only does it have the highest accuracy in the three models, but the
discriminant analysis is very simple and fast. However, the analysis is quite sensitive to outliers and
the size of the smallest group must be larger than the number of predictor variables. In addition,
50% accuracy isn’t very high. Therefore, there should be consideration of use of other models such
as the Support Vector Machine model which is a popular model since it uses little computation
power but still produces significant accuracy.
