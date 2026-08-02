# Continuous Authentication ML Project: A-to-Z Explanation

Source notebook: `NTI_ME (1) (1).ipynb`

## 1. Project Goal

The project uses behavioral and system information from user sessions to classify each session as:

- `Authorized`
- `Unauthorized`

This is a supervised binary classification problem. The model learns from historical sessions where the correct class is already known.

The project also studies which features are useful and checks whether the very high full-model performance depends strongly on application and command categories.

## 2. Dataset Overview

The dataset is `continuous_auth_dataset.csv`.

- Rows: `5,000`
- Columns: `19`
- Unique participants: `400`
- Unique sessions: `5,000`
- Missing values: `0`
- Exact duplicate rows: `0`
- Target balance: `2,750 Authorized`, `2,250 Unauthorized`

The target distribution is approximately:

- Authorized: `55%`
- Unauthorized: `45%`

## 3. Meaning of Every Column

| Column | Meaning | Use in final model |
|---|---|---|
| `Participant_ID` | Identifies the participant/user | Used only for grouping; not a predictive feature |
| `Session_ID` | Identifies one session | Removed because it is an identifier |
| `Timestamp` | Date and time of the session | Used to create time features; raw value removed |
| `Keyboard_Speed_WPM` | Typing speed in words per minute | Used |
| `Keystroke_Duration_ms` | Average key-hold duration in milliseconds | Used |
| `Typing_Accuracy_pct` | Typing accuracy percentage | Used |
| `Mouse_Speed_pxs` | Mouse movement speed in pixels per second | Used |
| `Active_Application` | Application active during the session | Used in the full model |
| `Language_Text` | Text associated with the session | Removed from model scope |
| `Command_Type` | Type of action, such as Search, Edit, or Download | Used in the full model |
| `Network_Status` | Network condition, such as Stable or Unstable | Used |
| `CPU_Usage_pct` | CPU usage percentage | Used |
| `Memory_Usage_pct` | Memory usage percentage | Used |
| `Previous_Authentication_Score` | Previous authentication-related score | Used, but should be confirmed as available before prediction |
| `Authentication_Score` | Current authentication score | Removed as a possible outcome-derived field |
| `Authentication_Label` | Authentication decision label | Removed as a possible leakage field |
| `Access_Action` | Action taken after authentication | Removed as a possible post-decision field |
| `Security_Level` | Security or confidence category | Removed as a possible outcome-derived field |
| `True_Class` | Correct class: Authorized or Unauthorized | Target column |

## 4. Data Loading and Validation

The notebook loads the CSV with Pandas:

```python
df = pd.read_csv(DATA_PATH)
```

It then converts the timestamp:

```python
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='raise')
```

The notebook validates:

- Missing values
- Duplicate rows
- Valid ranges for percentages and scores

The range checks include:

```python
Typing_Accuracy_pct: 0 to 100
CPU_Usage_pct: 0 to 100
Memory_Usage_pct: 0 to 100
Previous_Authentication_Score: 0 to 1
```

All implemented checks passed.

## 5. Exploratory Data Analysis

The notebook examines:

- Dataset shape
- First rows
- Data types
- Missing values
- Duplicate rows
- Target distribution
- Numerical feature distributions
- Class-wise numerical boxplots
- Categorical distributions
- Authentication-field relationships

The numerical values show strong class differences. For example:

| Feature | Authorized Mean | Unauthorized Mean |
|---|---:|---:|
| `Keyboard_Speed_WPM` | 54.69 | 40.90 |
| `Keystroke_Duration_ms` | 119.43 | 148.00 |
| `Typing_Accuracy_pct` | 92.82 | 84.26 |
| `Mouse_Speed_pxs` | 364.29 | 261.62 |
| `CPU_Usage_pct` | 36.55 | 57.55 |
| `Memory_Usage_pct` | 48.44 | 58.82 |
| `Previous_Authentication_Score` | 0.875 | 0.596 |

Some categorical values are strongly associated with the target. For example, in this dataset:

- `Browser`, `CMD`, `PowerShell`, and `Terminal` have 100% Unauthorized rates.
- `Excel`, `Outlook`, `Slack`, `Teams`, and `Word` have 0% Unauthorized rates.
- `Download`, `File Transfer`, and `Script Execution` have 100% Unauthorized rates.
- `Email`, `Edit`, `Chat`, and `Report` have 0% Unauthorized rates.

These patterns explain why the full model performs extremely well. They may reflect the synthetic or rule-generated nature of the dataset. They are not automatically leakage, but they should be mentioned as a limitation.

## 6. Removing Target and Unsafe Inputs

The target is separated:

```python
TARGET = 'True_Class'
y = df[TARGET].copy()
```

The project removes possible leakage and metadata columns:

```python
LEAKAGE_COLUMNS = [
    'Authentication_Score',
    'Authentication_Label',
    'Access_Action',
    'Security_Level'
]

IDENTIFIER_COLUMNS = ['Session_ID', 'Timestamp', 'Language_Text']
GROUP_COLUMN = 'Participant_ID'
```

The final predictive input matrix is created with:

```python
X = df.drop(columns=[TARGET, *LEAKAGE_COLUMNS, *IDENTIFIER_COLUMNS, GROUP_COLUMN])
```

`Participant_ID` is not used for prediction because the model should learn behavior rather than memorize user IDs. It is kept separately for grouping.

## 7. Participant-Aware Train/Test Split

The notebook uses:

```python
GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)
```

`Participant_ID` is passed as the group variable.

Actual split:

| Item | Value |
|---|---:|
| Training rows | 3,996 |
| Test rows | 1,004 |
| Training participants | 320 |
| Test participants | 80 |
| Overlapping participants | 0 |

This is important because sessions from the same participant can be similar. A row-level random split could allow the model to recognize a participant instead of learning general behavior.

## 8. Feature Engineering

The notebook creates simple features using `add_features`.

### `Hour`

Created from `Timestamp` using the hour of the day.

### `Is_Weekend`

Created from the timestamp. It is `1` for Saturday or Sunday and `0` otherwise.

### `Hour_Sin` and `Hour_Cos`

These represent the circular nature of time:

```python
Hour_Sin = sin(2 * pi * Hour / 24)
Hour_Cos = cos(2 * pi * Hour / 24)
```

Using both values helps the model understand that hour 23 and hour 0 are close in the daily cycle.

### `Keystroke_Efficiency`

```python
Keystroke_Duration_ms / Keyboard_Speed_WPM
```

This combines typing duration and typing speed.

### `Mouse_to_Keyboard_Ratio`

```python
Mouse_Speed_pxs / Keyboard_Speed_WPM
```

This represents the balance between mouse and keyboard activity.

### `Typing_Accuracy_Adjusted_Speed`

```python
Keyboard_Speed_WPM * Typing_Accuracy_pct / 100
```

This combines typing speed and accuracy.

The engineered features do not use `True_Class`, so their formulas do not directly leak the target.

## 9. Preprocessing Pipeline

The notebook separates numerical and categorical features:

```python
numeric_features = X_train.select_dtypes(include='number').columns.tolist()
categorical_features = X_train.select_dtypes(include='object').columns.tolist()
```

The preprocessing object is:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)
```

Numerical features use `StandardScaler`.

Categorical features use `OneHotEncoder(handle_unknown='ignore')`.

The preprocessing object is placed inside a scikit-learn `Pipeline`, so it is fitted only on the training data inside each cross-validation fold.

## 10. Models

The project compares four models:

### Dummy Baseline

```python
DummyClassifier(strategy='most_frequent')
```

This predicts the most common class and provides a simple reference point.

### Logistic Regression

```python
LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    random_state=42
)
```

This is a simple linear classification model. It benefits from numerical scaling.

### Random Forest

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    n_jobs=1
)
```

Random Forest combines many decision trees and can learn non-linear relationships.

### Gradient Boosting

```python
GradientBoostingClassifier(random_state=42)
```

Gradient Boosting builds trees sequentially so later trees improve earlier errors.

## 11. Cross-Validation and Model Selection

The project uses:

```python
GroupKFold(n_splits=3)
```

The grouping variable is `Participant_ID`.

The main CV results were:

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9915 | 0.9960 | 0.9850 | 0.9905 | 0.9989 |
| Gradient Boosting | 0.9900 | 0.9932 | 0.9844 | 0.9888 | 0.9996 |
| Logistic Regression | 0.9892 | 0.9922 | 0.9839 | 0.9880 | 0.9997 |
| Dummy Baseline | 0.5513 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

Random Forest was selected as the strongest full-feature comparison using grouped CV performance. The final project model is the tuned behavior-focused Random Forest, which excludes `Active_Application` and `Command_Type`.

## 12. Final Full-Feature Results

| Model | Training Accuracy | Test Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 1.0000 | 0.9940 | 1.0000 | 0.9869 | 0.9934 | 0.9992 |
| Gradient Boosting | 0.9922 | 0.9910 | 0.9956 | 0.9847 | 0.9901 | 0.9998 |
| Logistic Regression | 0.9912 | 0.9910 | 0.9934 | 0.9869 | 0.9901 | 0.9999 |
| Dummy Baseline | 0.5513 | 0.5448 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

The full-feature Random Forest comparison has:

- Test Accuracy: `99.40%`
- Test F1-score: `99.34%`
- Test ROC-AUC: `99.92%`
- Train-test accuracy gap: `1.0000 - 0.9940 = 0.0060`

## 13. Feature Ablation Experiment

Because `Active_Application` and `Command_Type` have strong rule-like relationships with the target, the project compares:

- Full features
- All features except `Active_Application` and `Command_Type`

Results:

| Experiment | CV F1 | CV ROC-AUC | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Features | 0.9905 | 0.9989 | 1.0000 | 0.9940 | 1.0000 | 0.9869 | 0.9934 | 0.9992 |
| Without Application and Command Features | 0.8586 | 0.9456 | 0.9995 | 0.8625 | 0.8900 | 0.7965 | 0.8406 | 0.9456 |

This shows that application and command features provide a large portion of the predictive signal. It does not prove that they are leakage. It does show that the dataset likely contains strong rule-generated relationships.

## 14. Reduced Behavior-Focused Model Tuning

The reduced model was tuned using a small, student-level Random Forest search. The excluded columns remained excluded:

```python
Active_Application
Command_Type
```

The selected tuned parameters were:

```python
max_depth=12
min_samples_split=10
min_samples_leaf=2
max_features='sqrt'
```

Before tuning:

- Training accuracy: `0.9995`
- Test accuracy: `0.8625`
- Test F1-score: `0.8406`
- Test ROC-AUC: `0.9456`
- Accuracy gap: `0.1369`

After tuning:

- Training accuracy: `0.9590`
- Test accuracy: `0.8665`
- Test F1-score: `0.8511`
- Test ROC-AUC: `0.9459`
- Accuracy gap: `0.0924`

Tuning reduced overfitting but did not remove it completely.

## 15. What the Project Proves

The project shows that:

1. Behavioral and system features can distinguish the two classes in this dataset.
2. Application and command categories are major sources of predictive signal.
3. A participant-aware split is important for behavioral authentication.
4. Preprocessing must be fitted inside a pipeline after splitting.
5. The reduced behavior-focused model remains useful but is weaker than the full model.
6. The dataset appears synthetic or rule-generated because several categories almost perfectly correspond to the target.

## 16. Project Limitations

- The dataset covers a limited synthetic-looking environment.
- Some categorical categories almost determine the target.
- The full model’s high scores may not transfer to real-world authentication data.
- `Previous_Authentication_Score` should be confirmed as available before the current prediction is made.
- The project does not include a production API, deployment architecture, or live monitoring system.
- Regression, anomaly detection, clustering, and SHAP are not part of the current final notebook workflow.

## 17. Simple Final Explanation

“This project classifies user sessions as Authorized or Unauthorized using behavioral and system features. I first inspected and validated the data, removed the target and possible leakage fields, and split the data by participant. I then created simple time and behavioral features and used a preprocessing pipeline with scaling and one-hot encoding.

I compared a baseline, Logistic Regression, Random Forest, and Gradient Boosting using participant-aware cross-validation. The full-feature Random Forest achieved 99.40% test accuracy, 99.34% F1-score, and 99.92% ROC-AUC, but the adopted final project model is the tuned behavior-focused Random Forest with 86.65% test accuracy, 85.11% F1-score, and 94.59% ROC-AUC.

Because application and command categories had very strong relationships with the target, I also tested a reduced behavior-focused model. Its performance dropped to 86.65% test accuracy and 85.11% F1-score after tuning, but it still performed better than the baseline. This shows that the behavioral features are useful, while the application and command fields provide major additional signal. The high results should be interpreted carefully because the dataset appears synthetic or rule-generated.”

