### Introduction	
Bank has multiple banking products that it sells to customer such as saving account, credit cards, investments etc. It wants to which customer will purchase its credit cards. For the same it has various kind of information regarding the demographic details of the customer, their banking behavior etc. Once it can predict the chances that customer will purchase a product, it wants to use the same to make pre-payment to the authors.

In this part I will demonstrate how to build a model, to predict which clients will subscribing to a term deposit, with inception of machine learning. In the ﬁrst part we will deal with the description and visualization of the analysed data, and in the second we will go to data classiﬁcation models. 

### Strategy

-Desire target\
-Data Understanding\
-Preprocessing Data\
-Machine learning Model\
-Prediction\
-Comparing Results

### Desire Target	
Predict if a client will subscribe (yes/no) to a term deposit — this is defined as a classification problem.

### Data 

The dataset (Assignment-2_data.csv) used in this assignment contains bank customers’ data.
File name: Assignment-2_Data
File format: . csv
Numbers of Row: 45212
Numbers of Attributes: 17 non- empty conditional attributes attributes and one decision attribute.

<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783430-eafd25b0-6d40-40b8-ac5b-1c4f67ca9e02.png">
<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783451-3e49b817-29a6-4108-b597-ce35897dda4a.png">

### Exploratory Data Analysis (EDA)


Data pre-processing is a main step in Machine Learning as the useful information which can be derived it from data set directly affects the model quality so it is extremely important to do at least necessary preprocess for our data before feeding it into our model.

In this assignment, we are going to utilize python to develop a predictive machine learning model. First, we will import some important and necessary libraries. 

Below we are can see that there are various numerical and categorical columns. The most important column here is y, which is the output variable (desired target): this will tell us if the client subscribed to a term deposit(binary: ‘yes’,’no’).

<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783456-78c22016-149b-4218-a4a5-765ca348f069.png">

We must to check missing values in our dataset if we do have any and do, we have any duplicated values or not.

<img width="115" alt="image" src="https://user-images.githubusercontent.com/91852182/143783471-a8656640-ec57-4f38-8905-35ef6f3e7f30.png"> 

We can see that in 'age' 9 missing values and 'balance' as well 3 values missed. In this case based that our dataset it has around 45k row I will remove them from dataset. on Pic 1 and 2 you will see before and after.

<img width="119" alt="image" src="https://user-images.githubusercontent.com/91852182/143783474-b3898011-98e3-43c8-bd06-2cfcde714694.png">

From the above analysis we can see that only 5289 people out of 45200 have subscribed which is roughly 12%. We can see that our dataset highly unbalanced. we need to take it as a note. 

<img width="101" alt="image" src="https://user-images.githubusercontent.com/91852182/143783534-a05020a8-611d-4da1-98cf-4fec811cb5d8.png">

Our list of categorical variables.

<img width="430" alt="image" src="https://user-images.githubusercontent.com/91852182/143783542-d40006cd-4086-4707-a683-f654a8cb2205.png">

Our list of numerical variables.

<img width="430" alt="image" src="https://user-images.githubusercontent.com/91852182/143783551-6b220f99-2c4d-47d0-90ab-18ede42a4ae5.png">

#### "Age" Q-Q Plots and Box Plot.

In above boxplot we can see that some point in very young age and as well impossible age. So, 

<img width="226" alt="image" src="https://user-images.githubusercontent.com/91852182/143783564-ad0e2a27-5df5-4e04-b5d7-6d218cabd405.png">
<img width="457" alt="image" src="https://user-images.githubusercontent.com/91852182/143783589-5abf0a0b-8bab-4192-98c8-d2e04f32a5c5.png">

Now, we don’t have issues on this feature so we can use it

<img width="230" alt="image" src="https://user-images.githubusercontent.com/91852182/143783599-5205eddb-a0f5-446d-9f45-cc1adbfcce67.png">
<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783601-e520d59c-3b21-4627-a9bb-cac06f415a1e.png">

#### "Duration" Q-Q Plots and Box Plot

<img width="231" alt="image" src="https://user-images.githubusercontent.com/91852182/143783634-03e5a584-a6fb-4bcb-8dc5-1f3cc50f9507.png">
<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783640-f6e71323-abbe-49c1-9935-35ffb2d10569.png">

This attribute highly affects the output target (e.g., if duration=0 then y=’no’). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.In this case I will not remove it we have very low 0. However, for realistic model we will need to place it to our depended and independent variables.

#### "Campaign" Q-Q Plots and Box Plot.

<img width="226" alt="image" src="https://user-images.githubusercontent.com/91852182/143783703-fe0b9f38-2414-4faa-95c8-a4fdbf61d612.png">
<img width="457" alt="image" src="https://user-images.githubusercontent.com/91852182/143783705-4ee13604-4a0b-4c9e-8943-b4b7cac35403.png">

I don’t see any outliers on this feature so we can use it without any preprocessing.

#### "Pdays" Q-Q Plots and Box Plot.

<img width="227" alt="image" src="https://user-images.githubusercontent.com/91852182/143783754-d3b642cc-22c1-47d6-936f-67e1cbbda52e.png">
<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783756-9d43923f-9d4d-44be-bd86-b590196c5983.png">

Number of days that passed by after the client was last contacted from a previous campaign (-1 means client was not previously contacted). We have to treat feature by using label encoding, because have -1 in 36940 values to mean client was not previously contacted.

#### "Previous" Q-Q Plots and Box Plot

<img width="227" alt="image" src="https://user-images.githubusercontent.com/91852182/143783763-0aaa2b1c-cdc3-46ae-be9d-be10447ce4dc.png">
<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783767-1971f8c5-0625-4cae-918d-68f2f13327c2.png">

Number of contacts performed before this campaign and for Particular client. Here we can see some outliers. We will clean them all.

<img width="223" alt="image" src="https://user-images.githubusercontent.com/91852182/143783785-d841f178-3433-45f4-ad5c-6e55c8c3e5ff.png">
<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783789-7064c864-8753-444d-a1e0-66283b48cbec.png">

It looks perfect now.

#### "Balance" Q-Q Plots and Box Plot.

<img width="217" alt="image" src="https://user-images.githubusercontent.com/91852182/143783795-5294a4ee-0673-4fc1-acda-09d05cb93c27.png">
<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783798-08eb5fbb-be2b-49d7-b713-5de15c7949dc.png">

All is clear he. We can proceed it without any changes.

#### "Day" Q-Q Plots and Box Plot.

<img width="215" alt="image" src="https://user-images.githubusercontent.com/91852182/143783808-30382bfd-476e-495a-a622-a9d76ceb2200.png">
<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783813-f067b060-7f5c-43d5-a3e4-9ef273e9995b.png">

All is clear he. We can proceed it without any changes.

#### Correlation. 

Correlation shows the relationship between variables in the dataset

<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783823-c65ce9f3-bb59-49b5-8068-d5c81049cbbe.png">

### Data Preprocessing

When building a machine learning model, it is important to preprocess the data to have an efficient model. 
We will need to change our 'pdays' to categorical data.
ML models are require all input and output values should to be numerical. So if our dataset have categorical data, we must have to encode it into the numbers before fit and evaluate a model. There are several methods available.Here I have used One-hot Encoding
Another data preprocessing method is to rescale our numerical columns; this helps to normalize our data within a particular range. Sklearn preprocessing StandardScaler() was used here.

<img width="210" alt="image" src="https://user-images.githubusercontent.com/91852182/143783845-2f883373-139b-4806-87d6-06ebb26c986b.png">

Output of data set after do the scaling.

Next, we will combine our tables. Frame with numerical columns which we scaled and normalize, and our categorical frame without original numerical data.

<img width="462" alt="image" src="https://user-images.githubusercontent.com/91852182/143783905-6d0c0efe-79cb-4359-9480-231b46928517.png">

To proceed in building our prediction model, we have to specify our dependent and independent variablels. Here we can place 'duration' for more realistic model.

By using below codes i have divide the data set into 30% for testing and 70% for training by using train_test_split from sklearn.model_selection. It is reasonable to always split the dataset into train and test set when building a machine learning model because it helps us to evaluate the performance of the model.

As you rememeber our data a bit imbalanced. This can affect our prediction. I will do oversampling here.

<img width="177" alt="image" src="https://user-images.githubusercontent.com/91852182/143783918-0a015178-fdca-44e9-8189-964b0a65f234.png">
<img width="177" alt="image" src="https://user-images.githubusercontent.com/91852182/143783920-e69bf697-46d3-49dd-9d6a-56c5ebf953bd.png">

Its applied-on training set. 
Now we a finally ready to do modeling and prediction. It always very important to preprocess data perfectly before jump to next step, we can see perfect result of our work

### Machine Learning Models and Predictions

I will first compare the model performance of the following 3 machine learning models using default hyperparameters:

•Logistic Regression\
•	Decision Tree\
•	K Nearest Neighbors (KNN)

First, we will load libraries which we will use for ML and plots with reports

#### Logistic Regression 

Logistic regression is a traditional machine learning model that fits a linear decision boundary between the positive and negative samples. Logsitic regression uses a line (Sigmoid function) in to predict if the dependent variable is true or false based on the independent variables. One advantage of logistic regression is the model is interpretable — we know which features are important for predicting positive or negative. Take note that the modeling is sensitive to the scaling of the features, so that is why we scaled the features above. We can fit logistic regression using the following code from scikit-learn

<img width="205" alt="image" src="https://user-images.githubusercontent.com/91852182/143783951-fe13fe5e-7f3c-45f1-aaf2-278af58de0c9.png">
<img width="240" alt="image" src="https://user-images.githubusercontent.com/91852182/143783955-1539234e-4e2b-4f61-af75-6c17d8d56c9d.png">

As you can see our accuracy is 0.90. From above code output we can see the overall prediction accuracy of the model. But we can’t evaluate the model by looking overall prediction accuracy only. So have to do the study with comparing to the classification report also.

#### Decision Tree		

This machine learning models is tree-based methods. The simplest tree-based method is known as a decision tree. The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules gotten from training data. In Decision Trees, for predicting a class label for a record we start from the root of the tree. One advantage of tree-based methods is that they have no assumptions about the structure of the data and are able to pick up non-linear effects if given sufficient tree depth. We can fit decision trees using the following code.

<img width="216" alt="image" src="https://user-images.githubusercontent.com/91852182/143783967-50bb32b2-c6e8-45ac-a83d-ccc62c7f1010.png">
<img width="239" alt="image" src="https://user-images.githubusercontent.com/91852182/143783970-c885bd0e-7221-4206-b5de-a04fceffa86d.png">

Our accuracy with this model 0.88. Let’s evaluate the model by looking overall prediction accuracy only. So have to do the study with comparing to the classification report as well.

#### K Nearest Neighbors (KNN) 	
	
KNN is one the simplest machine learning models. KNN looks at the k closest datapoints and probability sample that has positive labels. This model is very easy to understand, versatile, and you don’t need an assumption for the data structure. KNN is also good for multivariate analysis. A caveat with this algorithm is being sensitivity to K and takes a long time to evaluate if the number of trained samples is large. We can fit KNN using the following code from scikit-learn.

<img width="211" alt="image" src="https://user-images.githubusercontent.com/91852182/143783983-e729ad73-7733-4e5a-a8a7-8ab5982894d8.png">
<img width="240" alt="image" src="https://user-images.githubusercontent.com/91852182/143783988-86679a9c-0b44-4b81-b326-8bdaf8c3912a.png">

Our accuracy with this model 0.85. A bit lower than previous one. Let’s evaluate the model by looking overall prediction accuracy only. So have to do the study with comparing to the classification report as well.


<img width="494" alt="image" src="https://user-images.githubusercontent.com/91852182/143783996-28697e43-2474-459f-84e2-24bddee94383.png">

AUC (Area under the ROC Curve): It provides an aggregate measure of performance across all possible classification.

### Conclusion

<img width="770" alt="image" src="https://user-images.githubusercontent.com/91852182/143784174-ad259147-3f01-4146-bf6e-ba6abbceb101.png">


We were able to analyse bank marketing dataset, I built diﬀerent models which help us to analyse the dataset properly, I classify the dataset according to the data preparing
description. Here I showed various plots for easy reading and understanding. Result of my
classiﬁcation I present in the following table. I can see that obtain result of model mostly
are similar. But in my opinion the best one is Logistic regression model. It can predict the chances that customer will purchase a product across all possible classification with score 0.91






