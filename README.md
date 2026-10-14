# Credit Card Fraud Detection

## Project Overview

Credit card fraud detection is a classification problem where the challenge is not simply to identify fraudulent transactions, but to identify them without incorrectly flagging too many legitimate transactions.

I worked on this project to understand how machine learning models perform when fraudulent transactions represent only a very small proportion of the available data. The project focuses on exploring the dataset, preparing the features, comparing classification algorithms, and evaluating different approaches to handling class imbalance.

The analysis compares multiple models and resampling techniques to understand which approach provides the most useful balance between precision and recall.

#### Dataset

The project uses the European credit card transactions dataset covering two days in September 2013.

##### The dataset contains:

* Total transactions: 284,807
* Fraudulent transactions: 492
* Legitimate transactions: 284,315
* Fraudulent transactions: Approximately 0.172% of the dataset

##### The target variable is Class:

* 0 — Legitimate transaction
* 1 — Fraudulent transaction

Most of the transaction features (V1 to V28) are anonymized principal components obtained through Principal Component Analysis (PCA). The original Time and Amount features provide additional transaction information.

The dataset is highly imbalanced, which makes accuracy alone an unreliable measure of model performance.

Dataset availability: The transaction CSV is not included in this repository. Obtain the dataset from its original public source and place creditcard.csv in the project directory before running the notebook.

##### What I Explored

1. Exploratory Data Analysis

I started by examining the distribution of legitimate and fraudulent transactions and looking for differences in transaction amounts, timing, and feature distributions.

The analysis includes:

* Distribution of fraudulent versus legitimate transactions.
* Transaction amount distributions and outlier analysis.
* Fraud patterns across transaction time.
* Feature distributions for both classes.
* Correlation analysis using a heatmap.

This exploration helped me understand the extent of the class imbalance and identify features whose distributions appeared to overlap substantially between the two classes.

2. Feature Selection and Transformation

Based on the exploratory analysis, I removed selected features that showed substantial overlap between the classes, along with the original Time feature after deriving a time-based variable.

I also examined the skewness of transaction amounts. Since Amount had a strongly right-skewed distribution, I applied a Yeo-Johnson PowerTransformer and standardized the transformed values.

The transformation was fitted on the training data and then applied to the test data. This keeps the transformation parameters learned from the training set separate from the test set.

3. Training and Validation

I used a stratified train-test split with a 75% training set and a 25% test set.

The experiments use a fixed random state of 42 to make the split reproducible.

For model selection, I used five-fold StratifiedKFold cross-validation and GridSearchCV to evaluate parameter combinations.

#### The models were compared using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

I gave particular attention to precision, recall, and F1-score because missing fraudulent transactions and incorrectly flagging legitimate transactions have different consequences.

4. Classification Models

The notebook explores six classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree
5. Random Forest
6. XGBoost

Rather than relying on a single algorithm, I compared their cross-validation performance and investigated how their results changed under different class-balancing approaches.

5. Handling Class Imbalance

I evaluated the models using the original training distribution and experimented with three resampling techniques:

* Random Oversampling: Increases the representation of the minority class by resampling existing observations.
* SMOTE: Generates synthetic minority-class observations.
* ADASYN: Generates synthetic observations with greater emphasis on minority-class examples that are more difficult to learn.

The results showed that resampling did not improve every model. Its effect depended on the algorithm and the technique used.

This comparison was an important part of the project because balancing the classes does not automatically guarantee better fraud detection.

Results and Model Comparison

The following table contains the mean five-fold cross-validation results reported for the models trained without resampling.

Model	Accuracy	Precision	Recall	F1-score
Logistic Regression	0.9992	0.8906	0.6289	0.7333
K-Nearest Neighbors	0.9995	0.9257	0.7995	0.8577
Support Vector Machine	0.9995	0.9507	0.7289	0.8247
Decision Tree	0.9994	0.8970	0.7588	0.8199
Random Forest	0.9995	0.9297	0.7724	0.8430
XGBoost	0.9996	0.9434	0.8021	0.8666

These are cross-validation results, not test-set scores.

XGBoost achieved the highest cross-validated F1-score among these six models. KNN was also competitive, while Logistic Regression had a noticeably lower recall.

The high accuracy scores should be interpreted carefully. Because legitimate transactions dominate the dataset, a model can achieve very high accuracy while still missing a meaningful proportion of fraudulent transactions.

#### Test-Set Evaluation

The notebook reports a test-set F1-score of 0.8520 for the tuned XGBoost model.

The final Random Forest evaluation with Random Oversampling also reports a test-set F1-score of 0.8520.

For the Random Forest evaluation, the notebook reports the following confusion matrix:

Actual / Predicted	Legitimate	Fraud
Legitimate	71,074	5
Fraud	28	95

From this confusion matrix:

* True negatives: 71,074 legitimate transactions correctly classified.
* False positives: 5 legitimate transactions incorrectly flagged as fraud.
* False negatives: 28 fraudulent transactions missed.
* True positives: 95 fraudulent transactions correctly identified.

The results illustrate why precision and recall matter. A fraud detection model must identify suspicious transactions while keeping false alerts manageable.

The notebook also explores ROC curves and threshold selection. The final Random Forest evaluation identifies a probability threshold of 0.02 by maximizing the difference between the true positive rate and the false positive rate.

A threshold chosen this way is an analytical starting point, not necessarily the optimal threshold for a real financial institution. In practice, the decision should also consider fraud losses, investigation costs, and customer impact.

Key Findings

A few conclusions stood out during the experiments:

* XGBoost achieved the highest F1-score among the six models in the original-data cross-validation comparison.
* KNN also performed competitively, demonstrating that a simpler alternative can be worth evaluating.
* Resampling affected different algorithms differently; it did not consistently improve performance.
* Accuracy alone was not sufficient to judge model quality because of the extreme imbalance between legitimate and fraudulent transactions.
* The choice of probability threshold affects the trade-off between detecting more fraud and generating false alerts.

These findings reinforce the importance of evaluating a fraud detection model in the context of the problem it is intended to solve.

Tools and Libraries

The project uses:

* Python
* Pandas and NumPy for data manipulation
* Matplotlib and Seaborn for visualization
* Scikit-learn for preprocessing, classification, cross-validation, hyperparameter tuning, and evaluation
* Imbalanced-learn for Random Oversampling, SMOTE, ADASYN, and pipelines
* XGBoost for gradient-boosted classification
* Jupyter Notebook for experimentation

Running the Project

1. Clone the repository

git clone https://github.com/Byadab/Credit_card_fraudDetection.git
cd Credit_card_fraudDetection

2. Create a virtual environment

python3 -m venv .venv
source .venv/bin/activate

3. Install the required libraries

python -m pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter

4. Add the dataset

Place the downloaded creditcard.csv file in the project directory.

The current notebook uses a local absolute file path to load the CSV. Before running it on another computer, update that path to point to the location of your downloaded dataset.

5. Launch Jupyter Notebook

jupyter notebook

Open Credit_card_fraud_detection.ipynb and run the cells in order.

Limitations and Further Improvements

This project is an exploratory machine learning study rather than a production fraud detection system.

There are several areas worth improving:

* Replace the hardcoded dataset path with a portable relative path.
* Consolidate repeated modeling code into reusable pipelines.
* Ensure all preprocessing and feature selection steps are fitted within cross-validation folds to avoid information leakage.
* Reconcile model-specific test metrics and threshold-based evaluations in a single final comparison.
* Evaluate precision-recall curves and choose a threshold using explicit business costs.
* Test the selected model on additional data before considering real-world deployment.

These improvements would make the experiments easier to reproduce and the resulting model comparison more robust.

Project Files

File	Description
Credit_card_fraud_detection.ipynb	Main notebook containing exploratory analysis, feature preparation, model comparison, and evaluation
UG_Credit_card_fraud_detection_final.ipynb	Additional notebook version retained in the repository
Fraud_Detection_documentation.docx	Supporting project documentation

⸻

Project focus: Understanding class imbalance, comparing machine learning approaches, and evaluating fraud detection using metrics that reflect the cost of missed fraud and false alerts.
