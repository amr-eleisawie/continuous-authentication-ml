# Person 3 Machine Learning Explanation

This document explains only Person 3’s work in the current final notebook, in the required order:

1. Classification
2. Regression
3. Anomaly Detection
4. Evaluation
5. SHAP

The primary project task is binary classification of `True_Class` into `Authorized` and `Unauthorized`.

## 1. Classification

## What We Did

The project predicts whether a session is `Authorized` or `Unauthorized`. This is a supervised binary classification problem because the training data contains a known target column, `True_Class`.

Classification was selected because the project needs to make a class decision for each authentication session.

## Data Used

The final classification feature set contains:

- `Keyboard_Speed_WPM`
- `Keystroke_Duration_ms`
- `Typing_Accuracy_pct`
- `Mouse_Speed_pxs`
- `Active_Application`
- `Command_Type`
- `Network_Status`
- `CPU_Usage_pct`
- `Memory_Usage_pct`
- `Previous_Authentication_Score`
- `Hour`
- `Is_Weekend`
- `Hour_Sin`
- `Hour_Cos`
- `Keystroke_Efficiency`
- `Mouse_to_Keyboard_Ratio`
- `Typing_Accuracy_Adjusted_Speed`

The following fields were not used as predictive inputs:

- `True_Class`: the target column.
- `Authentication_Score`: removed because it may be generated from or after the authentication decision.
- `Authentication_Label`: an authentication outcome field and possible leakage source.
- `Access_Action`: an authentication decision field and possible leakage source.
- `Security_Level`: an authentication outcome or decision field and possible leakage source.
- `Session_ID`: a unique session identifier.
- `Timestamp`: used to create time features rather than passed as raw text.
- `Language_Text`: excluded from the project model scope.
- `Participant_ID`: kept separately for grouping, but not used as a predictive feature.

The target and leakage-sensitive columns are removed with:

```python
X = df.drop(columns=[TARGET, *LEAKAGE_COLUMNS, *IDENTIFIER_COLUMNS, GROUP_COLUMN])
y = df[TARGET].copy()
groups = df[GROUP_COLUMN].copy()
```

## Data Splitting

The notebook uses `GroupShuffleSplit`:

```python
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
```

The actual split was:

| Item | Result |
|---|---:|
| Training rows | 3,996 |
| Test rows | 1,004 |
| Training participants | 320 |
| Test participants | 80 |
| Overlapping participants | 0 |

The split is participant-aware because `Participant_ID` is passed as the grouping variable. This prevents sessions from the same participant appearing in both training and testing.

Stratification was not used. The project uses participant grouping as the more important requirement for behavioral authentication. The resulting class proportions remained reasonably similar: approximately 55.1% Authorized in training and 54.5% Authorized in testing.

## Models Used

The notebook compares four classification models.

### Dummy Baseline

`DummyClassifier(strategy='most_frequent')` predicts the most common class every time. It is used as a simple baseline to show what a model can achieve without learning feature patterns.

### Logistic Regression

`LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)` estimates the probability of each class using a linear relationship between the features and the target.

`class_weight='balanced'` gives more attention to the slightly less frequent class. Logistic Regression benefits from the `StandardScaler` used in the preprocessing pipeline.

### Random Forest

The main Random Forest uses:

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=1,
)
```

Random Forest combines many decision trees. Each tree learns simple decision rules, and the forest combines their predictions. It can model non-linear relationships between behavioral measurements and the target.

### Gradient Boosting

`GradientBoostingClassifier(random_state=42)` builds trees sequentially. Each new tree attempts to improve the mistakes made by the previous trees. It is useful for comparing another non-linear model with Random Forest.

## Problems and Solutions

### Problem

The original workflow contained preprocessing and possible leakage risks before the final pipeline was organized.

### Why It Was a Problem

Encoding, scaling, or using authentication-decision columns before a valid split could make the test results unreliable.

### How We Solved It

The final workflow removes the target and leakage-sensitive fields, uses a participant-aware split, and puts encoding and scaling inside the model `Pipeline`.

### Result After the Solution

The final notebook executed successfully. The participant overlap was zero, and the final models were evaluated using grouped cross-validation and a held-out participant-disjoint test set.

## Classification Results

| Model | Training Accuracy | Test Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 1.0000 | 0.9940 | 1.0000 | 0.9869 | 0.9934 | 0.9992 |
| Gradient Boosting | 0.9922 | 0.9910 | 0.9956 | 0.9847 | 0.9901 | 0.9998 |
| Logistic Regression | 0.9912 | 0.9910 | 0.9934 | 0.9869 | 0.9901 | 0.9999 |
| Dummy Baseline | 0.5513 | 0.5448 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

## What I Can Say During the Presentation

“Person 3’s main task was binary classification. We predicted whether each session was Authorized or Unauthorized. We removed the target, identifiers, raw text, and authentication decision fields from the model inputs. We used a participant-aware split, so the same participant did not appear in both training and testing. We compared a Dummy baseline, Logistic Regression, Random Forest, and Gradient Boosting. Random Forest was selected as the final model using grouped cross-validation. It achieved 99.40% test accuracy, 99.34% F1-score, and 99.92% ROC-AUC.”

## 2. Regression

Regression was not implemented in the provided project.

The final notebook focuses on classification of `True_Class`. No regression target, regression model, regression metrics, or regression results are included in the current final workflow.

## What I Can Say During the Presentation

“Regression was not implemented in the final version because the primary objective is classification of Authorized versus Unauthorized sessions. Therefore, there are no regression metrics or regression model results to report.”

## 3. Anomaly Detection

Anomaly detection was not implemented in the provided project.

The final notebook does not train an anomaly-detection algorithm or report anomaly flags. The project uses supervised classification because the target labels are available.

## What I Can Say During the Presentation

“Anomaly detection was not implemented in the final notebook. It would be different from classification because anomaly detection usually tries to identify unusual observations without directly using the known target labels. Our final project focuses on supervised classification instead.”

## 4. Evaluation

## Metrics Used

### Accuracy

Accuracy is the proportion of all test predictions that are correct. The selected Random Forest achieved test accuracy of `0.9940`, or 99.40%.

### Precision

Precision measures how many sessions predicted as `Unauthorized` were actually Unauthorized. The selected Random Forest achieved precision of `1.0000`.

### Recall

Recall measures how many actual Unauthorized sessions were detected. The selected Random Forest achieved recall of `0.9869`.

### F1-score

F1-score combines precision and recall into one value. It is useful when both types of classification error matter. The selected Random Forest achieved F1-score of `0.9934`.

### ROC-AUC

ROC-AUC measures how well the model ranks positive and negative examples across different probability thresholds. The selected Random Forest achieved ROC-AUC of `0.9992`.

### Confusion Matrix and Classification Report

The notebook generates a confusion matrix for the selected model using:

```python
confusion = confusion_matrix(
    y_test,
    best_predictions,
    labels=['Authorized', 'Unauthorized']
)
```

It also prints a classification report containing precision, recall, F1-score, and support for both classes. The saved final notebook does not include a separate numeric confusion-matrix table in its stored output, so exact cell counts are not reported here.

## Cross-Validation

The project uses grouped cross-validation:

```python
GroupKFold(n_splits=3)
```

The groups are `Participant_ID`. This is important because ordinary row-level folds could place sessions from the same participant in both the training and validation portions.

The main cross-validation results were:

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9915 | 0.9960 | 0.9850 | 0.9905 | 0.9989 |
| Gradient Boosting | 0.9900 | 0.9932 | 0.9844 | 0.9888 | 0.9996 |
| Logistic Regression | 0.9892 | 0.9922 | 0.9839 | 0.9880 | 0.9997 |
| Dummy Baseline | 0.5513 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

## Overfitting Analysis

For the selected Random Forest:

- Training accuracy: `1.0000`
- Test accuracy: `0.9940`
- Train-test accuracy gap: `0.0060`

The training accuracy is perfect, but the test accuracy is also very high and close to it. This indicates a small generalization gap in the final full-feature model, not severe overfitting based on accuracy.

The reduced behavior-focused model had a larger gap before simple tuning. Its training accuracy was `0.9995`, while test accuracy was `0.8625`. After simple tuning, the reduced model’s training accuracy became `0.9590` and test accuracy became `0.8665`, reducing the gap from `0.1369` to `0.0924`. This reduced-model experiment is an additional analysis; the primary full-feature model remained the main final model.

## Final Model Selection

The full-feature Random Forest was the strongest comparison model using grouped cross-validation. The adopted final project model is the tuned behavior-focused Random Forest, which excludes `Active_Application` and `Command_Type`.

The selection considered the grouped CV results, with F1-score considered first and ROC-AUC used as an additional comparison measure. The held-out test set was then used for the final report.

The full-feature Random Forest had the highest CV F1-score among the compared full-feature models. The final project decision additionally considers realism and adopts the tuned behavior-focused model. Logistic Regression and Gradient Boosting had slightly higher ROC-AUC values in the displayed results, but their test F1 and accuracy were slightly lower.

## Problems and Solutions

### Problem

The full-feature model has extremely high performance, and some categorical features have very strong rule-like relationships with `True_Class`.

### Why It Was a Problem

This may make the dataset easier than a real-world authentication dataset. High performance alone does not prove leakage, but it limits how confidently the results can be generalized.

### How We Solved It

The project performed a feature-ablation comparison and trained a reduced behavior-focused model without `Active_Application` and `Command_Type`. This did not replace the main model, but it measured how much performance depended on those fields.

### Result After the Solution

The reduced model remained useful but was weaker. After simple tuning, it achieved test accuracy `0.8665`, F1-score `0.8511`, and ROC-AUC `0.9459`. This shows that behavioral and system features contain useful information, while the application and command categories contribute major predictive signal.

## What I Can Say During the Presentation

“I evaluated the models using accuracy, precision, recall, F1-score, ROC-AUC, a confusion matrix, and a classification report. I also used three-fold GroupKFold cross-validation based on Participant ID. The final Random Forest had training accuracy of 1.0000 and test accuracy of 0.9940, so the accuracy gap was only 0.0060. I selected it using grouped cross-validation, not the test results. I also checked the high performance with a reduced-feature experiment, which showed that the application and command features are major sources of signal.”

## 5. SHAP

SHAP explainability was not implemented in the provided project.

The current final notebook does not import SHAP, create a SHAP explainer, calculate SHAP values, or display a SHAP summary plot.

## What I Can Say During the Presentation

“SHAP explainability was not implemented in the final version of the project. The project currently reports model performance and feature-group ablation, but it does not provide individual SHAP feature explanations.”

## Final Summary

| Task | What Person 3 Did | Main Problem | How It Was Solved | Final Result |
|---|---|---|---|---|
| Classification | Trained and compared Dummy, Logistic Regression, Random Forest, and Gradient Boosting models | Possible over-optimistic performance from rule-like categorical relationships | Removed outcome fields, used participant-aware splitting, grouped CV, and feature ablation | Random Forest: 99.40% test accuracy, 99.34% F1, 99.92% ROC-AUC |
| Regression | Not implemented in the current final notebook | No regression workflow exists | Not applicable | No regression result reported |
| Anomaly Detection | Not implemented in the current final notebook | No anomaly workflow exists | Not applicable | No anomaly result reported |
| Evaluation | Used grouped CV, test metrics, classification report, and confusion matrix code | Need to avoid selecting using the test set | Selected using grouped CV and used test data for final reporting | Strong full-feature generalization with 0.0060 accuracy gap |
| SHAP | Not implemented in the current final notebook | No explainability workflow exists | Not applicable | No SHAP result reported |

## Complete Presentation Script

“My part of the project focused mainly on classification and model evaluation.

For classification, the goal was to predict whether a session was Authorized or Unauthorized using the target `True_Class`. The model inputs included behavioral features, system features, network status, application and command information, previous authentication score, and simple engineered features. I removed the target, identifiers, raw text, and authentication-decision fields that could cause leakage.

For splitting, I used a participant-aware split with `GroupShuffleSplit`. There were 3,996 training rows and 1,004 test rows. The training set contained 320 participants and the test set contained 80 participants. The overlap was zero, so the same participant was not present in both sets.

I compared a Dummy baseline, Logistic Regression, Random Forest, and Gradient Boosting. The Dummy model gave a reference point. Logistic Regression provided a simple linear model. Random Forest and Gradient Boosting were used to model non-linear relationships. The final Random Forest had test accuracy of 99.40%, F1-score of 99.34%, and ROC-AUC of 99.92%.

Regression was not implemented in the final notebook because the main project objective is classification. Anomaly detection was also not implemented. It would be a different unsupervised problem from the supervised classification task.

For evaluation, I used grouped three-fold cross-validation, accuracy, precision, recall, F1-score, ROC-AUC, a confusion matrix, and a classification report. Random Forest was selected using grouped cross-validation instead of using the test set to choose the model. Its training accuracy was 1.0000 and test accuracy was 0.9940, giving a small accuracy gap of 0.0060.

The dataset has very strong relationships between some categorical features and the target, so I also checked a reduced behavior-focused model. Removing application and command features reduced performance, but the reduced model still performed meaningfully. This suggests that the behavioral and system features contain useful information, while the categorical fields provide major additional signal.

SHAP explainability was not implemented in the final notebook. Therefore, the final project uses standard evaluation metrics and feature-group ablation rather than SHAP plots.

Overall, the final classification workflow is participant-aware, uses grouped validation, and reports multiple metrics. The main limitation is that the dataset appears easier and more rule-based than a real-world authentication dataset.”

## Likely Instructor Questions

### 1. Why is this a classification problem?

Because the target `True_Class` contains two categories: `Authorized` and `Unauthorized`.

### 2. Why did you use `Participant_ID` for grouping?

Sessions from the same participant may be behaviorally similar. Grouping prevents the same participant from appearing in both training and testing.

### 3. Was stratification used?

No. The project used `GroupShuffleSplit` to enforce participant separation. The resulting train and test class proportions were still reasonably similar.

### 4. What is data leakage?

Data leakage occurs when target information or test information enters model training. It can produce unrealistically high results.

### 5. Why were authentication decision columns removed?

`Authentication_Score`, `Authentication_Label`, `Access_Action`, and `Security_Level` may be generated from or after the authentication decision. Using them could make the model learn the answer directly.

### 6. Why did you include a Dummy Classifier?

It provides a simple baseline. A real model should perform better than a model that always predicts the most common class.

### 7. Why was Random Forest selected?

It had the strongest grouped CV F1-score and strong test accuracy and F1-score. It also models non-linear relationships between the features and the target.

### 8. Does training accuracy of 1.0000 prove overfitting?

No. It suggests the model fits the training data perfectly, but the test accuracy was 0.9940, so the gap was only 0.0060. The test result must also be considered.

### 9. Why is F1-score useful here?

F1-score combines precision and recall. Both missing Unauthorized sessions and incorrectly flagging Authorized sessions matter in authentication.

### 10. What does ROC-AUC measure?

ROC-AUC measures how well the model separates the two classes across different probability thresholds.

### 11. Why was GroupKFold used?

It keeps participant groups separated between the training and validation portions. This gives a more realistic estimate for unseen participants.

### 12. Did the model use the test set to select the final model?

No. The final selection used grouped cross-validation. The test set was used afterward for final reporting.

### 13. Was regression implemented?

No. Regression was not implemented in the provided project. The primary task is classification of `True_Class`.

### 14. Was anomaly detection implemented?

No. Anomaly detection was not implemented in the provided project. It would be a separate unsupervised task.

### 15. Was SHAP implemented?

No. SHAP explainability was not implemented in the provided project. The project used standard metrics and feature ablation instead.

