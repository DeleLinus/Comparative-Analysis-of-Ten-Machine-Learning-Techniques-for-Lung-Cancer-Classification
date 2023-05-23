<h1 align=center>Comparative Analysis of Ten Machine Learning Techniques for Lung Cancer Classification</h1>
<h2 align=center>Project Overview</h2>
This project compare 10 machine learning algorithms including Neural Networks in diagnosing lung cancer in patients based on variables defining the patient, including age, allergy, and whether the patient smokes. The data dictionary showing the columns of the data set and their descriptions is shown in <a href="##data_dict">Data Description</a>

**Research questions:**

1. Which machine learning model yields the highest classification accuracy for lung cancer diagnosis using the given dataset?
2. How do different machine learning models perform in terms of precision, recall, and F1-score in the classification of lung cancer?
3. Which features (or combinations of features) are most important for each machine learning model in classifying lung cancer?


This project is a typical supervised learning task and a typical  binary classification problem as it involves predicting between two classes (cancerous, non-cancerous). More specifically, this is a multivariate classification problem since the system will use multiple features to make prediction.
<br>
Machine Learning Classification Algorithms used include:
LogisticRegression 
from sklearn.neighbors import 
from sklearn.ensemble import 
from sklearn.ensemble import GradientBoostingClassifier
* Support Vector Machine (SVM)
* Logistic Regression
* K Nearest Neighbors Classifier
* Naive Bayes
* Linear Discriminant Analysis
* Decission Tree
* Ensemble learning algorithms 
    - Random Forest Classifier
    - GradientBoostingClassifier
    - BaggingClassifier
* Artificial Neural Network (ANN) using Keras

 **Performance Measure:**
 80% of the dataset is used for model training while the remaining 20% is used as a test set to evaluate the models.
 
  
 **Evaluation Metric**
 The choice of metric for a classification machine learning problem typically depends on the **business/live implication** and properties of the dataset used in training the model. Hence, since the dataset is imbalanced, the following metrics are used:
* Accuracy
* Precision
* Recall
* Specificity
* Balanced Accuracy
* F1 score

<br>The balanced accuracy is used to measure the overall capacity of the models.<br>


<h2 id="data_dict">Data Description</h2>
It it important to reiterate that I did not create this dataset, it is a widely used Kaggle data. The data format is CSV (comma-separated values) and consists of 309 rows, 16 columns comprising quantitative variables, and 15 qualitative and discrete quantitative variables, including the target variable. These variables define the patient, including age, allergy, and whether the patient smokes.

**GENDER :** Sex of the patient - M [Male] , F [Female]

**AGE :** Age of patients in years (Quantitative)

**SMOKING :** Whether the patient smokes or not - 2 [Yes] , 1 [No]

**YELLOW_FINGERS :** 2 [Yes] , 1 [No]\\

**ANXIETY :** 2 [Yes] , 1 [No]

**PEER_PRESSURE :** 2 [Yes] , 1 [No]

**CHRONIC DISEASE :** 2 [Yes] , 1 [No]

**FATIGUE :** 2 [Yes] , 1 [No]

**ALLERGY :** 2 [Yes] , 1 [No]

**WHEEZING :** 2 [Yes] , 1 [No]

**ALCOHOL CONSUMING :** 2 [Yes] , 1 [No]

**COUGHING :** 2 [Yes] , 1 [No]

**SHORTNESS OF BREATH :** 2 [Yes] , 1 [No]

**SWALLOWING DIFFICULTY :** 2 [Yes] , 1 [No]

**CHEST PAIN :** 2 [Yes] , 1 [No]

**LUNG_CANCER :** YES [Positive] , NO [Negative]
