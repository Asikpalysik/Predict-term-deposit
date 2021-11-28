# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as stats
import pylab
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")

# Import raw data
dataset = pd.read_csv("Assignment-2_data.csv")
dataset.name = "dataset"
dataset.head()

# We need to check missing values in dataset
dataset.isnull().sum()

# To check duplicated values
print(dataset.duplicated().value_counts())  ### we dont have

# Remove row of missing value
dataset.dropna(inplace=True)

# Lets count numbers of row for 'y' type
dataset.groupby("y").size()

# We will any way need a list of numeric data
dataset._get_numeric_data().columns.tolist()

# List of categorical variables
categorical_data = dataset.select_dtypes(exclude="number")
categorical_data.head()

# List of numerical variables
numerical_data = dataset.select_dtypes(include="number")
numerical_data.head()

# Outliers and Feature Q-Q Plots and Box Plot (age)
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(dataset["age"], dist="norm", plot=plt)
plt.show()

plt.rcParams["figure.figsize"] = (22, 3)
temp_df = pd.DataFrame(dataset, columns=["age"])
temp_df.boxplot(vert=False)

## Clearing data
dataset = dataset[dataset["age"] >= 18]
dataset = dataset[dataset["age"] < 120]

## Checking plots
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(dataset["age"], dist="norm", plot=plt)
plt.show()

plt.rcParams["figure.figsize"] = (22, 3)
temp_df = pd.DataFrame(dataset, columns=["age"])
temp_df.boxplot(vert=False)

# Outliers and Feature Q-Q Plots and Box Plot (duration)
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(dataset["duration"], dist="norm", plot=plt)
plt.show()

plt.rcParams["figure.figsize"] = (22, 3)
temp_df = pd.DataFrame(dataset, columns=["duration"])
temp_df.boxplot(vert=False)

# Outliers and Feature Q-Q Plots and Box Plot (Campaign)
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(dataset["campaign"], dist="norm", plot=plt)
plt.show()

plt.rcParams["figure.figsize"] = (22, 3)
temp_df = pd.DataFrame(dataset, columns=["campaign"])
temp_df.boxplot(vert=False)

# Outliers and Feature Q-Q Plots and Box Plot (Pdays)
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(dataset["pdays"], dist="norm", plot=plt)
plt.show()

plt.rcParams["figure.figsize"] = (22, 3)
temp_df = pd.DataFrame(dataset, columns=["pdays"])
temp_df.boxplot(vert=False)

len(dataset[dataset["pdays"] == -1])  # -> 36940

# Outliers and Feature Q-Q Plots and Box Plot (Previous)
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(dataset["previous"], dist="norm", plot=plt)
plt.show()

plt.rcParams["figure.figsize"] = (22, 3)
temp_df = pd.DataFrame(dataset, columns=["previous"])
temp_df.boxplot(vert=False)

## Clearing data
dataset = dataset[dataset["previous"] <= 50]
dataset = dataset.reset_index(drop=True)

## Checking plots
plt.rcParams["figure.figsize"] = (22, 3)
temp_df = pd.DataFrame(dataset, columns=["previous"])
temp_df.boxplot(vert=False)

plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(dataset["previous"], dist="norm", plot=plt)
plt.show()

# Outliers and Feature Q-Q Plots and Box Plot (Balance)
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(dataset["balance"], dist="norm", plot=plt)
plt.show()

plt.rcParams["figure.figsize"] = (22, 3)
temp_df = pd.DataFrame(dataset, columns=["balance"])
temp_df.boxplot(vert=False)

# Outliers and Feature Q-Q Plots and Box Plot (day)
plt.rcParams["figure.figsize"] = (10, 6)
stats.probplot(dataset["day"], dist="norm", plot=plt)
plt.show()

plt.rcParams["figure.figsize"] = (22, 3)
temp_df = pd.DataFrame(dataset, columns=["day"])
temp_df.boxplot(vert=False)

# Correlation matrix
fig, ax = plt.subplots()
fig.set_size_inches(20, 10)
ax = sns.heatmap(dataset.corr(), annot=True, fmt=".1g", vmin=-1, vmax=1, center=0)
plt.title("Correlation", y=1)
plt.xlabel("Features")
plt.ylabel("Features")

# List categorical columns
cat_cols = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "pdays",
]

# list numerical columns
num_cols = ["duration", "balance", "campaign", "day", "age", "previous"]

# Making pdays as Categorical (-1 no call)
dataset.pdays = pd.Categorical(dataset.pdays)

# Y into 1 and 0
dataset["y"] = (dataset.y == "yes").astype("int")
# Dummies
dataset = pd.get_dummies(dataset)
dataset.head()

# Import library for rescaling
from sklearn.preprocessing import StandardScaler

# Rescale our numerical columns
scaler = StandardScaler()
scaler.fit(dataset[num_cols])
data_scaled = scaler.transform(dataset[num_cols])
columns_value_new = dataset[num_cols].columns
data_scaled_ok = pd.DataFrame(data_scaled, columns=columns_value_new)
data_scaled_ok.head(10)

# Now we need to combite all tables
# We will remove not nessasaries num cols
data1 = dataset.drop(num_cols, axis=1)
result = pd.concat([data_scaled_ok, data1], axis=1, join="inner")
display(result)

# Chosing target and dropin target and not impotant variables
data = result.drop(columns=["y"])
target = result.filter(["y"], axis=1)

# Split data on 70/30
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=10
)

# Plot to check inbalance
sns.countplot(x="y", data=result)
plt.show()

# Importing the necessary function
from imblearn.over_sampling import SMOTE

# Creating an instance
sm = SMOTE(random_state=27)
# Applying it to the training set
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

# Recheck inbalance
sns.countplot(x="y", data=y_train_smote)
plt.show()

# We will need to drop first ID to have more clear
# prediction and in the end keep it back
X_train_smote_no_ID = X_train_smote.drop(["Id"], axis=1)
X_test_no_ID = X_test.drop(["Id"], axis=1)

# Import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve

# 1 MLM Logistic Regression
lreg = LogisticRegression()
lreg.fit(X_train_smote_no_ID, y_train_smote)
prediction_1 = lreg.predict(X_test_no_ID)

# Score report
print(classification_report(y_test, prediction_1))

# Plot
plot_1 = plot_confusion_matrix(
    lreg,
    X_test_no_ID,
    y_test,
    display_labels=["NO", "YES"],
    cmap=plt.cm.Reds,
    values_format=".2f",
)
plot_1.figure_.suptitle("Confusion Matrix")
plt.show()

# 2 MLM Decision tree
clf = DecisionTreeClassifier(max_depth=10, random_state=40)
clf.fit(X_train_smote_no_ID, y_train_smote)
prediction_2 = clf.predict(X_test_no_ID)

# Score report
print(classification_report(y_test, prediction_2))

# Plot
plot_2 = plot_confusion_matrix(
    clf,
    X_test_no_ID,
    y_test,
    display_labels=["NO", "YES"],
    cmap=plt.cm.Reds,
    values_format=".2f",
)
plot_2.figure_.suptitle("Confusion Matrix")
plt.show()

# 3 MLM KNN
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train_smote_no_ID, y_train_smote)
prediction_3 = knn.predict(X_test_no_ID)

# Score report
print(classification_report(y_test, prediction_3))

# Plot
plot_3 = plot_confusion_matrix(
    knn,
    X_test_no_ID,
    y_test,
    display_labels=["NO", "YES"],
    cmap=plt.cm.Reds,
    values_format=".2f",
)
plot_3.figure_.suptitle("Confusion Matrix")
plt.show()

# ROC plot
classifiers = [lreg, clf, knn]
ax = plt.gca()
for i in classifiers:
    plot_roc_curve(i, X_test_no_ID, y_test, ax=ax)

# Final file
result = pd.DataFrame()
result["Id"] = X_test["Id"]
result["Y"] = prediction_1
result["Y"].replace(0, "no", inplace=True)
result["Y"].replace(1, "yes", inplace=True)
result.to_csv("result file.csv", header=True, index=False)

