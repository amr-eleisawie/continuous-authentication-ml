# Person 1 Data Preprocessing Explanation

This document explains only the preprocessing work in the current notebook:

1. Data Cleaning
2. Missing Values
3. Outliers
4. Encoding
5. Scaling

The primary classification target is `True_Class`. The project uses a participant-aware train/test split before the final preprocessing pipeline is fitted.

## 1. Data Cleaning

### What the dataset looked like before cleaning

The notebook loads `continuous_auth_dataset.csv` using `pandas.read_csv`. The dataset contains 5,000 rows and 19 columns. It includes participant and session identifiers, a timestamp, numerical behavioral/system measurements, categorical fields, authentication-related fields, and the target `True_Class`.

The original timestamp is read as text and is converted later using:

```python
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='raise')
```

### Checks and cleaning performed

The notebook checks for exact duplicate rows:

```python
assert df.duplicated().sum() == 0, 'Duplicate rows require review before modeling.'
```

The output shows:

```text
Exact duplicate rows: 0
```

No duplicate rows were removed because none were found.

The notebook also checks that the timestamp can be converted successfully. In addition, it validates expected ranges for selected numerical columns:

```python
range_checks = {
    'Typing_Accuracy_pct': (0, 100),
    'CPU_Usage_pct': (0, 100),
    'Memory_Usage_pct': (0, 100),
    'Previous_Authentication_Score': (0, 1),
}

for column, (lower, upper) in range_checks.items():
    assert df[column].between(lower, upper).all(), f'Invalid values found in {column}.'
```

The range checks passed. No invalid values were reported.

The project later removes columns that are not used as classification inputs:

```python
LEAKAGE_COLUMNS = [
    'Authentication_Score', 'Authentication_Label',
    'Access_Action', 'Security_Level'
]
IDENTIFIER_COLUMNS = ['Session_ID', 'Timestamp', 'Language_Text']
GROUP_COLUMN = 'Participant_ID'

X = df.drop(columns=[TARGET, *LEAKAGE_COLUMNS, *IDENTIFIER_COLUMNS, GROUP_COLUMN])
```

This removes the target, authentication decision fields, identifiers, raw timestamp, and raw text from the model feature matrix. `Participant_ID` is retained separately as the grouping variable for the participant-aware split.

### Problems Found

- No exact duplicate rows were found.
- No invalid values were found by the implemented range checks.
- The timestamp initially has to be converted from text to datetime.
- Some columns are not appropriate model inputs because they are targets, possible leakage fields, identifiers, or raw text.

### How We Solved Them

- Converted `Timestamp` with `pd.to_datetime(..., errors='raise')`.
- Checked duplicate rows with `df.duplicated()`.
- Checked expected numerical ranges with `Series.between()`.
- Removed target, leakage-sensitive, identifier, timestamp, and raw-text columns from `X`.
- Kept `Participant_ID` separately as `groups` for the split.

### What I Can Say During the Presentation

“First, I loaded the dataset and checked its basic structure. I converted the timestamp to the correct datetime type, checked for duplicate rows, and validated important numerical ranges. The dataset had no duplicate rows and no invalid values according to these checks. I then separated the target and removed identifiers, raw text, and authentication-decision columns from the model inputs. I kept the participant ID separately because it was needed to prevent the same participant from appearing in both training and testing.”

## 2. Missing Values

### What was found

The notebook checks missing values with:

```python
assert df.isna().sum().sum() == 0, 'Missing values require review before modeling.'
```

The actual result was:

```text
Missing values: 0
```

Therefore, no column contained missing values in the provided dataset.

### Imputation and indicators

No missing values were removed or filled. No missing-value indicators were created because there were no missing observations to represent.

Median, mean, or mode imputation was not needed in the final notebook.

### Pipeline and leakage

There is no fitted imputer in the final preprocessing pipeline because the dataset contains no missing values. Other learned preprocessing operations are placed inside the modeling `Pipeline` and are fitted after the train/test split.

### Problems Found

- No missing-value problem was found in the provided dataset.
- No imputation problem exists in the final notebook because imputation was unnecessary.

### How We Solved Them

- Verified that the total number of missing values was zero.
- Continued without imputation or missing-value indicators.

### What I Can Say During the Presentation

“I checked all columns for missing values. The result was zero missing values, so I did not remove rows or fill values artificially. Because there were no missing values, an imputer and missing-value indicators were not necessary. The remaining learned preprocessing steps were still placed inside the model pipeline so that they would be fitted only on training data.”

## 3. Outliers

### Columns checked

The final notebook checks domain ranges for these numerical columns:

- `Typing_Accuracy_pct`
- `CPU_Usage_pct`
- `Memory_Usage_pct`
- `Previous_Authentication_Score`

Other numerical behavioral columns are included in the model, but the final cleaning section does not report a separate IQR or Z-score outlier count for them.

### Method used

The final notebook uses domain range checks, not an IQR or Z-score outlier-removal procedure. It does not claim that statistical outliers were removed or capped.

The relevant code is:

```python
for column, (lower, upper) in range_checks.items():
    assert df[column].between(lower, upper).all(), f'Invalid values found in {column}.'
```

### Results and decision

All implemented range checks passed. The notebook does not report a numerical count of IQR or Z-score outliers, and it does not remove, cap, or transform extreme observations using those methods.

The project therefore treats values inside the accepted domain ranges as valid observations. This is reasonable for a student project because a high or low behavioral measurement is not automatically an error.

### Problems Found

- A complete statistical outlier analysis using IQR or Z-score was not implemented in the final notebook.
- No IQR/Z-score outlier counts are available from the final workflow.

### How We Solved Them

- Applied domain range validation to the selected constrained numerical columns.
- Kept observations that satisfied the expected ranges.
- Did not claim that all statistical outliers were removed.

### What I Can Say During the Presentation

“For outliers, I used simple domain checks rather than automatically deleting extreme values. For example, accuracy, CPU usage, memory usage, and previous authentication score must stay within their valid ranges. All of these checks passed. I did not remove every unusual value because an unusual behavioral measurement may still be a real observation. The final notebook does not perform IQR or Z-score capping.”

## 4. Encoding

### Categorical columns

The final model contains categorical columns such as:

- `Active_Application`
- `Command_Type`
- `Network_Status`

The notebook identifies categorical features from the training data:

```python
categorical_features = X_train.select_dtypes(include='object').columns.tolist()
```

### Method used

The final pipeline uses `OneHotEncoder`:

```python
OneHotEncoder(handle_unknown='ignore')
```

One-hot encoding creates a separate numerical indicator column for each category. It does not impose an artificial ranking such as saying one application is greater than another.

`handle_unknown='ignore'` means that if a category appears in test data but was not present in the training data, the encoder does not crash. It represents that unseen category with zeros for the known category columns.

### Pipeline and leakage

Encoding is inside the `ColumnTransformer`, which is inside the model `Pipeline`:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', StandardScaler(), numeric_features),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)
```

The pipeline is fitted using training data during model training and cross-validation. This prevents category information from the held-out test set from being used to fit the encoder in advance.

The final notebook does not use `LabelEncoder` for the input categorical features.

### Problems Found

- Categorical input values cannot be used directly by the scikit-learn numerical models.
- A category may appear in a future or test partition that was not seen during training.

### How We Solved Them

- Used `OneHotEncoder` for the categorical input features.
- Used `handle_unknown='ignore'` for unseen categories.
- Put encoding inside the training pipeline so it is fitted after splitting.

### What I Can Say During the Presentation

“The dataset contains categorical features such as the active application, command type, and network status. Machine learning models need numerical inputs, so I converted these categories with one-hot encoding. I used `handle_unknown='ignore'` so an unseen test category would not cause an error. The encoder is inside the pipeline, so it learns categories from the training data only and avoids data leakage.”

## 5. Scaling

### Numerical columns

The final notebook automatically identifies the numerical training columns:

```python
numeric_features = X_train.select_dtypes(include='number').columns.tolist()
```

These include behavioral, system, historical, and engineered numerical features such as:

- `Keyboard_Speed_WPM`
- `Keystroke_Duration_ms`
- `Typing_Accuracy_pct`
- `Mouse_Speed_pxs`
- `CPU_Usage_pct`
- `Memory_Usage_pct`
- `Previous_Authentication_Score`
- `Hour`
- `Is_Weekend`
- `Hour_Sin`
- `Hour_Cos`
- The simple behavioral ratio features

### Method used

The project uses `StandardScaler`:

```python
StandardScaler()
```

Standardization transforms numerical features so that they are centered around a mean of zero with a standard deviation of approximately one, based on the training data.

Scaling is especially useful for Logistic Regression because it makes numerical feature magnitudes more comparable. It is less important for tree-based models such as Random Forest and Gradient Boosting, but using one consistent preprocessing pipeline keeps the model comparison simple and fair.

### Pipeline and leakage

Scaling is inside the `ColumnTransformer`, which is inside each model `Pipeline`. Therefore, the scaler is fitted only on the training portion in each cross-validation fold and then applied to validation or test data.

The project does not fit `StandardScaler` on the complete dataset before splitting in the final workflow.

### Problems Found

- Numerical features have different units and scales, such as percentages, pixels per second, milliseconds, and scores.
- Tree models do not require scaling, but Logistic Regression benefits from it.

### How We Solved Them

- Applied `StandardScaler` to numerical features.
- Applied scaling through the `ColumnTransformer` and `Pipeline`.
- Fitted the scaler only on training data during model fitting and cross-validation.

### What I Can Say During the Presentation

“The numerical features use different units. For example, keystroke duration is measured in milliseconds, mouse speed uses pixels per second, and other features are percentages or scores. I used `StandardScaler` so the numerical values are on comparable scales. This is important for Logistic Regression. It is not required for tree models, but keeping the same pipeline makes the model comparison consistent. The scaler is fitted only on training data, which avoids leakage.”

## Final Summary

| Stage | What We Found | What We Did | Why It Was Important |
|---|---|---|---|
| Data Cleaning | 5,000 rows and 19 columns; zero duplicate rows; valid checked ranges; timestamp required conversion | Converted `Timestamp`, checked duplicates and ranges, and separated model features from target, identifiers, raw text, and leakage-sensitive fields | Made the data usable and prevented invalid or inappropriate inputs from entering the model |
| Missing Values | Total missing values: 0 | No imputation or missing-value indicators were needed | Avoided unnecessary changes to complete data |
| Outliers | Domain range checks passed; no IQR/Z-score treatment was implemented | Kept valid observations and did not claim that statistical outliers were removed | Prevented automatic deletion of potentially valid behavior measurements |
| Encoding | `Active_Application`, `Command_Type`, and `Network_Status` are categorical | Used `OneHotEncoder(handle_unknown='ignore')` inside the pipeline | Converted categories to numerical model inputs safely |
| Scaling | Numerical features use different units | Used `StandardScaler` inside the pipeline | Helped scale-sensitive models and avoided train/test leakage |

## Person 1 Complete Presentation Script

“My part of the project focused on data preprocessing. I followed five steps: data cleaning, missing values, outliers, encoding, and scaling.

First, I loaded the continuous authentication dataset. It contains 5,000 rows and 19 columns. I converted the `Timestamp` column from text into a datetime value. I also checked for exact duplicate rows, and the result was zero. I checked valid ranges for features such as typing accuracy, CPU usage, memory usage, and previous authentication score. These checks passed.

After that, I separated the target and model inputs. The target is `True_Class`. I removed identifiers, raw text, and authentication decision fields from the model feature matrix. I kept `Participant_ID` separately because it was needed for the participant-aware split.

For missing values, I checked the complete dataset and found zero missing values. Because the dataset was complete, I did not remove rows, fill values, or create missing-value indicators. An imputer was not necessary.

For outliers, I used simple domain validation. I checked that percentage features stayed between zero and one hundred and that the authentication score stayed between zero and one. The checks passed. I did not automatically remove every extreme behavioral value because an unusual value may still be a valid user behavior. The final workflow does not use IQR or Z-score capping.

Next, I handled categorical features. The input categories include `Active_Application`, `Command_Type`, and `Network_Status`. I used one-hot encoding so that each category became a numerical indicator. I used `handle_unknown='ignore'` so that an unseen category would not break the test prediction step.

Finally, I scaled the numerical features using `StandardScaler`. The features have different units, so scaling is useful especially for Logistic Regression. Scaling is less important for tree models, but the same pipeline makes the comparison consistent.

The important point is that encoding and scaling are inside the preprocessing pipeline. They are fitted only on training data during the split and cross-validation. This prevents information from the test data from influencing the learned preprocessing steps.

After these steps, the data was clean, numerical features were prepared, categorical features were encoded, and the model was ready for training.”

## Important Questions the Instructor May Ask

### 1. Why did you handle missing values?

Missing values can cause errors or reduce model quality. In this dataset, the missing-value count was zero, so no imputation was needed.

### 2. Why did you not fill missing values with the mean or median?

There were no missing values to fill. Applying imputation unnecessarily would change complete data without a reason.

### 3. What is an outlier?

An outlier is an observation that is unusually high or low compared with the rest of the data. It is not automatically an error; it may represent valid behavior.

### 4. Did you remove all outliers?

No. The final notebook used domain range checks and did not implement IQR or Z-score removal. The checked values were inside their valid ranges.

### 5. Why do we encode categorical data?

Most scikit-learn models need numerical inputs. Encoding converts categories into numerical columns that the models can process.

### 6. Why did you choose One-Hot Encoding?

One-hot encoding represents categories without creating a false numerical order between them. This is appropriate for application names, command types, and network statuses.

### 7. What does `handle_unknown='ignore'` do?

It allows the model to process a category that was not seen during training without raising an error.

### 8. Why do we scale numerical features?

The numerical columns use different units. Scaling makes them more comparable and is helpful for models such as Logistic Regression.

### 9. What is data leakage?

Data leakage happens when information from the target or test data influences training. It can make the evaluation look better than real performance.

### 10. Why should preprocessing be fitted on training data only?

The test set should represent unseen data. If the scaler or encoder learns from the test set, the evaluation is no longer completely fair. The pipeline fits preprocessing on training data only.

