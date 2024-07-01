# Wine Quality Prediction using Support Vector Machine

This project aims to predict the quality of white wine based on various chemical properties using Support Vector Machine (SVM) as a classifier. The dataset used for this project is sourced from the UCI Machine Learning Repository.

## Dataset

The dataset contains 4898 instances of white wine with 12 variables:

1. `fixed acidity`
2. `volatile acidity`
3. `citric acid`
4. `residual sugar`
5. `chlorides`
6. `free sulfur dioxide`
7. `total sulfur dioxide`
8. `density`
9. `pH`
10. `sulphates`
11. `alcohol`
12. `quality`

The target variable is `quality`, which is a score between 0 and 10.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- pandas
- numpy
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn
```

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### Dataset

The dataset is directly loaded from the following URL:

```python
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/WhiteWineQuality.csv', sep=';')
```

### Exploratory Data Analysis

To get a basic understanding of the dataset, the following steps were performed:

1. **First Five Rows of DataFrame:**

   ```python
   df.head()
   ```

2. **Information of DataFrame:**

   ```python
   df.info()
   ```

3. **Summary Statistics:**

   ```python
   df.describe()
   ```

4. **Unique Values in the Target Variable:**

   ```python
   df['quality'].value_counts()
   ```

5. **Grouping by Quality:**

   ```python
   df.groupby('quality').mean()
   ```

### Data Preprocessing

The features were standardized using `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X = ss.fit_transform(X)
```

### Train-Test Split

The dataset was split into training and testing sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=2529)
```

### Model Training

A Support Vector Machine (SVM) model was trained on the training set:

```python
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
```

### Model Evaluation

The model's performance was evaluated using a confusion matrix and classification report:

```python
from sklearn.metrics import confusion_matrix, classification_report

y_pred = svc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### Binary Classification

For binary classification, wine quality was labeled as `1` if the quality score was 6 or higher, otherwise as `0`:

```python
y = df['quality'].apply(lambda y_value: 1 if y_value >= 6 else 0)
```

### Future Prediction

To make future predictions, a sample from the dataset was used:

```python
df_new = df.sample(1)
X_new = df_new.drop(['quality'], axis=1)
X_new = ss.transform(X_new)
svc.predict(X_new)
```

## Results

The model achieved an accuracy of approximately 78% in predicting the binary classification of wine quality.

## Conclusion

This project demonstrates how SVM can be used to predict wine quality based on its chemical properties. Further improvements can be made by tuning the hyperparameters of the SVM model or by using different classification algorithms.

## Author

Pravesh Kumar


