# Person 2 Data Analysis Explanation

This document explains only Person 2’s work in the current final notebook:

1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. Feature Selection
4. Clustering

The primary target in the project is `True_Class`, with the classes `Authorized` and `Unauthorized`.

## 1. Exploratory Data Analysis (EDA)

## Dataset Overview

The dataset is `continuous_auth_dataset.csv`. It contains session-level continuous authentication information. Each row represents one session and contains behavioral measurements, system measurements, contextual categories, authentication-related fields, and the true class.

The dataset contains 5,000 rows and 19 columns. There are 400 unique participants and 5,000 unique sessions.

Important identifier and time columns:

- `Participant_ID`: identifies the participant and is later used for grouped splitting.
- `Session_ID`: identifies a session.
- `Timestamp`: records the session time.

Numerical features:

- `Keyboard_Speed_WPM`
- `Keystroke_Duration_ms`
- `Typing_Accuracy_pct`
- `Mouse_Speed_pxs`
- `CPU_Usage_pct`
- `Memory_Usage_pct`
- `Previous_Authentication_Score`
- `Authentication_Score`

Categorical or text fields:

- `Active_Application`
- `Language_Text`
- `Command_Type`
- `Network_Status`
- `Authentication_Label`
- `Access_Action`
- `Security_Level`
- `True_Class`

The target used for classification is `True_Class`.

## Initial Data Exploration

The notebook checked the dataset shape, displayed the first rows, inspected data types, checked missing values, checked duplicate rows, and displayed the target distribution.

The main outputs were:

```text
Dataset shape: (5000, 19)
Missing values: 0
Exact duplicate rows: 0
Unique participants: 400
Unique sessions: 5000
```

The first rows showed the expected behavioral and contextual columns, including `Keyboard_Speed_WPM`, `Keystroke_Duration_ms`, `Typing_Accuracy_pct`, `Mouse_Speed_pxs`, `Active_Application`, `Command_Type`, and `Network_Status`.

The notebook also displayed summary statistics and category counts during the EDA work. The dataset has 10 active-application categories, 8 command-type categories, and 2 network-status categories.

## Target Analysis

The target is `True_Class`. It contains two classes:

```text
Authorized      2750
Unauthorized    2250
```

The percentages are:

```text
Authorized      55.0%
Unauthorized    45.0%
```

This is a reasonably balanced binary target. It is not a severe class-imbalance problem, although both classes still need to be evaluated separately. Accuracy alone would not be enough for an authentication project, so precision, recall, F1-score, and ROC-AUC are also used later.

The target distribution was visualized with a Seaborn count plot. The purpose was to check whether one class dominated the dataset and whether class weighting or special sampling might be necessary.

## Numerical Feature Analysis

The numerical behavioral and system features were examined using boxplots and summary statistics. The main numerical values have different units and ranges:

- Keyboard speed is measured in WPM.
- Keystroke duration is measured in milliseconds.
- Typing accuracy is a percentage.
- Mouse speed is measured in pixels per second.
- CPU and memory usage are percentages.
- Previous authentication score is between 0 and 1.

The class-wise analysis of the dataset showed meaningful differences between Authorized and Unauthorized sessions:

| Feature | Authorized Mean | Unauthorized Mean | Difference (Unauthorized - Authorized) |
|---|---:|---:|---:|
| `Keyboard_Speed_WPM` | 54.69 | 40.90 | -13.79 |
| `Keystroke_Duration_ms` | 119.43 | 148.00 | 28.56 |
| `Typing_Accuracy_pct` | 92.82 | 84.26 | -8.55 |
| `Mouse_Speed_pxs` | 364.29 | 261.62 | -102.66 |
| `CPU_Usage_pct` | 36.55 | 57.55 | 21.00 |
| `Memory_Usage_pct` | 48.44 | 58.82 | 10.38 |
| `Previous_Authentication_Score` | 0.875 | 0.596 | -0.279 |

The simple correlations with a binary Unauthorized indicator were:

| Feature | Correlation |
|---|---:|
| `Previous_Authentication_Score` | -0.675 |
| `Typing_Accuracy_pct` | -0.519 |
| `CPU_Usage_pct` | 0.518 |
| `Keyboard_Speed_WPM` | -0.495 |
| `Mouse_Speed_pxs` | -0.473 |
| `Keystroke_Duration_ms` | 0.456 |
| `Memory_Usage_pct` | 0.394 |

These are associations, not proof of causation. They show that the behavioral and system measurements contain useful predictive information in this dataset.

## Categorical Feature Analysis

The notebook examined the distributions of `Active_Application`, `Command_Type`, and `Network_Status`.

The application distribution included:

- `Chrome`: 18.48%
- `Teams`: 9.40%
- `Slack`: 9.38%
- `CMD`: 9.36%
- `Browser`: 9.32%
- `Excel`: 9.14%
- `Outlook`: 9.06%
- `Terminal`: 9.04%
- `Word`: 8.52%
- `PowerShell`: 8.30%

The command-type distribution included:

- `Search`: 27.28%
- `Email`: 16.98%
- `Download`: 16.00%
- `Script Execution`: 10.98%
- `Edit`: 10.96%
- `File Transfer`: 6.62%
- `Report`: 5.70%
- `Chat`: 5.48%

For `Network_Status`:

- `Stable`: 66.34%
- `Unstable`: 33.66%

The dataset also shows very strong target-rate differences by category. For example:

- `Browser`, `CMD`, `PowerShell`, and `Terminal` have a 100% Unauthorized rate.
- `Excel`, `Outlook`, `Slack`, `Teams`, and `Word` have a 0% Unauthorized rate.
- `Download`, `File Transfer`, and `Script Execution` have a 100% Unauthorized rate.
- `Email`, `Edit`, `Chat`, and `Report` have a 0% Unauthorized rate.
- `Unstable` has a 74.21% Unauthorized rate, while `Stable` has a 30.18% Unauthorized rate.

These relationships are extremely strong and appear rule-like. They are not automatically leakage, but they explain why the full-feature model performs much better than the reduced behavior-focused model. The project should describe this as a limitation of the synthetic or rule-generated dataset.

## Correlation Analysis

The original EDA code included a correlation heatmap for the selected numerical features. Correlation was used to inspect relationships and possible redundancy between numerical variables.

The later dataset audit quantified relationships with the target. The strongest numerical association was for `Previous_Authentication_Score`, followed by typing accuracy, CPU usage, keyboard speed, mouse speed, and keystroke duration.

Correlation does not prove that one feature causes the target. It only measures an association. The categorical target-rate analysis was also important because a standard numerical correlation matrix cannot fully describe categorical relationships.

The current final notebook does not use a formal correlation threshold to automatically remove features.

## EDA Visualizations

Important plots used in the project include:

### Target count plot

This plot shows the number of Authorized and Unauthorized rows. It was created to check class balance. It showed a 55%/45% split, so the dataset was reasonably balanced.

### Numerical boxplots by class

The boxplots compare numerical feature values between Authorized and Unauthorized sessions. They help reveal differences in spread, center, and unusual values. The later class-wise results confirmed meaningful differences in typing speed, keystroke duration, typing accuracy, mouse speed, CPU usage, memory usage, and previous authentication score.

### Categorical count plots

The count plots show how often each application, command type, and network status occurs. They were used to understand category frequency and identify rare or dominant categories.

### Correlation heatmap

The heatmap shows pairwise numerical correlations. It was created to inspect relationships and possible redundancy. The project did not use the heatmap as an automatic feature-removal rule.

## Problems Found During EDA

Actual findings include:

- The dataset is synthetic-looking because several categorical values almost perfectly determine the target class.
- Numerical behavioral and system features are strongly associated with the target.
- The target is not perfectly balanced, although the 55%/45% split is reasonable.
- The full model’s high performance is strongly influenced by `Active_Application` and `Command_Type`.

### How the Problems Were Handled

The project did not remove features only because they were predictive. Instead, it performed a feature ablation experiment. Removing `Active_Application` and `Command_Type` reduced Random Forest test F1 from `0.9934` to `0.8406`, showing that these fields are major predictive signals.

The reduced model was then tuned with simple Random Forest settings. The tuned reduced model achieved:

- Test Accuracy: `0.8665`
- Test F1-score: `0.8511`
- Test ROC-AUC: `0.9459`

This confirms that the behavioral and system features still contain useful information, although the full feature set performs much better on this dataset.

### What I Can Say During the Presentation

“In the EDA stage, I first checked the dataset structure, class distribution, numerical features, categorical features, and relationships between features. The target had 55% Authorized and 45% Unauthorized records, so it was reasonably balanced. The behavioral features showed clear differences between the two classes. I also found very strong rule-like relationships for some application and command categories. I did not automatically remove those features, so I later used feature ablation to measure how much they affected the model.”

## 2. Feature Engineering

Feature engineering was implemented in the function `add_features`.

```python
def add_features(features, timestamps):
    result = features.copy()
    timestamp_values = pd.to_datetime(timestamps)
    result['Hour'] = timestamp_values.dt.hour.to_numpy()
    result['Is_Weekend'] = (timestamp_values.dt.dayofweek >= 5).astype(int).to_numpy()
    result['Hour_Sin'] = np.sin(2 * np.pi * result['Hour'] / 24)
    result['Hour_Cos'] = np.cos(2 * np.pi * result['Hour'] / 24)
    speed = result['Keyboard_Speed_WPM'].clip(lower=1e-6)
    result['Keystroke_Efficiency'] = result['Keystroke_Duration_ms'] / speed
    result['Mouse_to_Keyboard_Ratio'] = result['Mouse_Speed_pxs'] / speed
    result['Typing_Accuracy_Adjusted_Speed'] = (
        result['Keyboard_Speed_WPM'] * result['Typing_Accuracy_pct'] / 100
    )
    return result
```

### `Hour`

**Created from:** `Timestamp`

**Formula or logic:** `timestamp_values.dt.hour`

**Why it was created:** To represent the hour of day when a session occurred.

**What it represents:** Possible time-of-day behavior patterns.

**Expected benefit:** User behavior may differ during working hours and non-working hours.

**Actual result:** The feature was included in the final feature set. Its simple target correlation was weak, approximately `0.036`.

### `Is_Weekend`

**Created from:** `Timestamp`

**Formula or logic:** One when the day is Saturday or Sunday, otherwise zero.

**Why it was created:** To represent a possible weekday/weekend behavior difference.

**What it represents:** Whether a session occurred during the weekend.

**Expected benefit:** Weekend activity could differ from normal working behavior.

**Actual result:** Its simple target correlation was very weak, approximately `-0.020`.

### `Hour_Sin`

**Created from:** `Hour`

**Formula or logic:** `sin(2 * pi * Hour / 24)`

**Why it was created:** To represent the circular nature of time. Hour 23 and hour 0 are close in time even though their integer difference is large.

**What it represents:** The cyclic position of the session within a 24-hour day.

**Expected benefit:** It avoids treating the beginning and end of the day as far apart.

**Actual result:** Its simple target correlation was weak, approximately `-0.026`.

### `Hour_Cos`

**Created from:** `Hour`

**Formula or logic:** `cos(2 * pi * Hour / 24)`

**Why it was created:** It works with `Hour_Sin` to represent the complete circular time pattern.

**What it represents:** The second cyclic component of the session time.

**Expected benefit:** It gives the model a more complete representation of daily time cycles.

**Actual result:** Its simple target correlation was weak, approximately `-0.017`.

### `Keystroke_Efficiency`

**Created from:** `Keystroke_Duration_ms` and `Keyboard_Speed_WPM`

**Formula or logic:** `Keystroke_Duration_ms / Keyboard_Speed_WPM`

The code clips keyboard speed to `1e-6` as a safety measure against division by zero.

**Why it was created:** To combine typing duration and typing speed into one behavioral measure.

**What it represents:** A simple relationship between how long keystrokes take and how fast the user types.

**Expected benefit:** It may distinguish different typing styles better than either original feature alone.

**Actual result:** It showed a moderate target relationship, approximately `0.509` correlation with the Unauthorized indicator.

### `Mouse_to_Keyboard_Ratio`

**Created from:** `Mouse_Speed_pxs` and `Keyboard_Speed_WPM`

**Formula or logic:** `Mouse_Speed_pxs / Keyboard_Speed_WPM`

**Why it was created:** To describe the balance between mouse activity and keyboard activity.

**What it represents:** Whether a session is relatively more mouse-oriented or keyboard-oriented.

**Expected benefit:** Different users or session types may have different interaction balances.

**Actual result:** Its simple target correlation was weak, approximately `-0.056`.

### `Typing_Accuracy_Adjusted_Speed`

**Created from:** `Keyboard_Speed_WPM` and `Typing_Accuracy_pct`

**Formula or logic:** `Keyboard_Speed_WPM * Typing_Accuracy_pct / 100`

**Why it was created:** To combine typing speed and accuracy.

**What it represents:** Fast and accurate typing receives a higher value than fast but inaccurate typing.

**Expected benefit:** It may provide a more meaningful measure of effective typing behavior.

**Actual result:** It had a moderate negative target relationship, approximately `-0.526` correlation with the Unauthorized indicator.

## Feature Engineering Problems

- Some engineered features, especially time features, showed weak direct relationships with the target.
- Ratio features can amplify noise when a denominator is very small, although the code uses `clip(lower=1e-6)` for keyboard speed.
- The project did not perform a separate formal ablation for every individual engineered feature.
- The engineered features do not use `True_Class`, so there is no direct target leakage in their formulas.
- `Timestamp` must be available at prediction time for the time-based features to be usable.

### How the Problems Were Handled

- The features were kept simple and based on understandable behavioral logic.
- A small epsilon guard was used for division-based features.
- The full versus reduced feature experiments were used to understand the effect of feature groups.
- The project kept the main classification pipeline and did not add complex feature-generation methods.

### What I Can Say During the Presentation

“I created simple features from time and behavioral measurements. The time features describe hour, weekend status, and the circular position of the hour. The behavioral features combine typing speed, keystroke duration, mouse speed, and typing accuracy. These features were created using only input information, not the target. Some time features were weak individually, but they were kept because they are easy to explain and could represent real behavior patterns.”

## 3. Feature Selection

### What was implemented

The project used domain-based feature selection rather than a formal statistical feature-selection algorithm.

The target and clearly inappropriate columns were removed before modeling:

```python
TARGET = 'True_Class'
LEAKAGE_COLUMNS = [
    'Authentication_Score', 'Authentication_Label',
    'Access_Action', 'Security_Level'
]
IDENTIFIER_COLUMNS = ['Session_ID', 'Timestamp', 'Language_Text']
GROUP_COLUMN = 'Participant_ID'

X = df.drop(columns=[TARGET, *LEAKAGE_COLUMNS, *IDENTIFIER_COLUMNS, GROUP_COLUMN])
y = df[TARGET].copy()
groups = df[GROUP_COLUMN].copy()
```

The model candidate features were the original behavioral, system, historical, and contextual inputs, followed by the engineered features.

The final feature list included:

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

### What was removed and why

- `True_Class` was removed because it is the target.
- `Authentication_Score`, `Authentication_Label`, `Access_Action`, and `Security_Level` were removed because they may be generated from or after an authentication decision.
- `Session_ID` was removed because it is a unique identifier, not behavioral information.
- `Timestamp` was removed as a raw input, then used to create deterministic time features.
- `Language_Text` was removed because it was considered outside the project’s model scope and could contain sensitive text information.
- `Participant_ID` was not used as a predictive input. It was retained separately for participant-aware grouping.

### Methods actually used

The project did not use `SelectKBest`, mutual information, Recursive Feature Elimination, or an automatic model-based selector.

Correlation analysis was used for understanding relationships, but not as an automatic removal rule. The feature list was selected mainly using domain knowledge and leakage prevention.

### Feature Selection Problems

- Formal feature selection was not implemented in the provided project.
- Several categorical fields have extremely strong rule-like relationships with `True_Class`.
- No individual feature-selection method was used to prove that every retained feature was necessary.
- The final feature choice was made using domain decisions and group-aware model comparisons rather than a formal selector.

### How the Problems Were Handled

- Obvious target, identifier, raw-text, and possible post-decision columns were removed.
- Participant ID was kept only for grouping and not for prediction.
- A feature ablation experiment compared the full model with a model that removed only `Active_Application` and `Command_Type`.
- The full model remained the main model because the experiment showed predictive value, not confirmed leakage.

### What I Can Say During the Presentation

“Feature selection was mainly based on domain knowledge and leakage prevention. I removed the target, identifiers, raw text, and authentication-decision fields that could make the evaluation unfair. I did not use an advanced automatic feature selector. Instead, I compared the full feature set with a reduced set to understand the contribution of the application and command features.”

## 4. Clustering

### Clustering Objective

Clustering is not implemented in the current final notebook. The current notebook explicitly states that regression, anomaly detection, and clustering were removed from the main workflow and can be added later as separate experiments.

Therefore, there is no clustering objective, clustering algorithm, cluster-count selection, cluster assignment table, or cluster interpretation in the current final project workflow.

### Data Preparation for Clustering

Not implemented in the provided project.

The final notebook does not prepare a clustering matrix, scale a clustering-specific feature set, or calculate cluster labels.

### Clustering Algorithm

Not implemented in the provided project.

### Choosing the Number of Clusters

Not implemented in the provided project.

### Clustering Results

Not implemented in the provided project.

### Cluster Interpretation

Not implemented in the provided project. The classification labels `Authorized` and `Unauthorized` should not be described as clusters because they are supervised target classes.

### Clustering Problems

The only project-level limitation is that clustering is not part of the current final notebook. No incorrect clustering result was produced, so there is no unsupported cluster interpretation to correct.

### How the Problems Were Handled

Clustering was clearly separated from the primary classification workflow rather than being presented as a completed result. The project focuses on supervised classification of `True_Class`.

### What I Can Say During the Presentation

“Clustering was not implemented in the final version of the notebook. The main project objective is supervised classification of Authorized and Unauthorized sessions. I did not present the classification labels as clusters, because clusters are unsupervised groups and are different from known target classes.”

## Final Summary

| Stage | What We Did | Main Finding | Problem Found | How We Handled It |
|---|---|---|---|---|
| EDA | Inspected shape, rows, types, missing values, duplicates, target distribution, numerical features, categories, and relationships | The dataset is reasonably class-balanced, but several categorical fields have rule-like target relationships | Very strong category-target associations and synthetic-looking structure | Measured the effect with feature ablation instead of removing features automatically |
| Feature Engineering | Created time features and simple behavioral ratios/interactions | Behavioral combinations contained useful information; some time features were weak individually | Ratios can amplify noise and timestamp features require prediction-time availability | Used simple formulas and a small epsilon guard; kept the pipeline readable |
| Feature Selection | Removed the target, possible leakage fields, identifiers, raw timestamp input, raw text, and participant ID from predictive inputs | The final feature set combines behavioral, system, network, historical, and contextual information | No formal automatic feature selector was implemented | Used domain knowledge, leakage checks, and feature-group comparison |
| Clustering | Not implemented in the current final notebook | No clustering result is claimed | Clustering was outside the final primary workflow | Clearly marked as not implemented and kept separate from supervised classification |

## Person 2 Complete Presentation Script

“My part of the project focused on data analysis, feature engineering, feature selection, and clustering.

First, I performed exploratory data analysis. The dataset is called `continuous_auth_dataset.csv`, and it contains 5,000 rows and 19 columns. Each row represents a session with behavioral features such as keyboard speed, keystroke duration, typing accuracy, and mouse speed. It also contains system features such as CPU and memory usage, categorical context such as active application and command type, and the target `True_Class`.

I checked the dataset shape, first rows, data types, missing values, duplicate rows, unique values, and target distribution. There were no missing values and no duplicate rows. The target contained 2,750 Authorized rows and 2,250 Unauthorized rows, so the classes were reasonably balanced.

I analyzed numerical features with summary statistics and boxplots. I found clear differences between the two target classes. For example, Authorized sessions had higher average keyboard speed, typing accuracy, and mouse speed. Unauthorized sessions had higher average keystroke duration, CPU usage, and memory usage. Previous authentication score also had a strong relationship with the target.

I also analyzed categorical features. The most important finding was that some application and command categories were almost perfectly associated with one class. This explains why the full model is very accurate. I did not automatically remove these features because high predictive power alone does not prove leakage. Instead, I compared a full model with a reduced model later.

For feature engineering, I created simple time and behavioral features. The time features were `Hour`, `Is_Weekend`, `Hour_Sin`, and `Hour_Cos`. The behavioral features were `Keystroke_Efficiency`, `Mouse_to_Keyboard_Ratio`, and `Typing_Accuracy_Adjusted_Speed`. These features combine existing measurements in ways that are understandable for a continuous authentication project. The formulas do not use the target, so they do not directly introduce target leakage.

For feature selection, I removed the target, identifiers, raw text, and authentication decision fields from the predictive inputs. I kept `Participant_ID` separately for grouping, but I did not use it as a model feature. I did not use an advanced automatic selector. The decisions were based on domain knowledge and leakage prevention. I also used feature ablation to understand how important the application and command features were.

Clustering was not implemented in the current final notebook. The primary project task is supervised classification of `True_Class`, so I did not confuse the known target labels with unsupervised clusters. If clustering is added later, it should be treated as a separate exploratory experiment.

Overall, Person 2’s analysis showed that the dataset has useful behavioral information but also has very strong rule-like categorical relationships. The feature engineering and selection steps prepared the data for the modeling stage while keeping the workflow simple and understandable.”

## Important Questions the Instructor May Ask

### 1. What is EDA?

EDA means Exploratory Data Analysis. It is the process of understanding the dataset using summaries, counts, plots, and relationships before training models.

### 2. Why did you analyze the target distribution?

To check whether one class dominated the data. The target was 55% Authorized and 45% Unauthorized, so it was reasonably balanced.

### 3. What did the numerical analysis show?

Authorized and Unauthorized sessions had different average behavioral and system measurements. For example, they differed in keyboard speed, typing accuracy, mouse speed, keystroke duration, and CPU usage.

### 4. What did the categorical analysis show?

Some application and command categories had almost exclusively one target class. This is a strong predictive pattern and may reflect the synthetic design of the dataset.

### 5. What did the correlation analysis show?

The strongest numerical relationship with the Unauthorized indicator was for `Previous_Authentication_Score`, followed by typing accuracy, CPU usage, keyboard speed, mouse speed, and keystroke duration. Correlation shows association, not causation.

### 6. What is feature engineering?

Feature engineering means creating useful input features from existing columns. In this project, examples include cyclic time features and simple behavioral ratios.

### 7. Why did you create `Hour_Sin` and `Hour_Cos`?

Because time is circular. Hour 23 and hour 0 are close in the real day, so sine and cosine represent that relationship better than using only an integer hour.

### 8. How do you know whether a feature is useful?

We can compare model performance with and without the feature or feature group. In this project, the feature ablation experiment measured the effect of `Active_Application` and `Command_Type`.

### 9. What is feature selection?

Feature selection is deciding which columns should be model inputs. In this project, the target, identifiers, raw text, and possible post-decision fields were removed using domain knowledge and leakage prevention.

### 10. What is clustering, and did you use it?

Clustering is an unsupervised method for grouping similar observations without using a target label. Clustering was not implemented in the current final notebook, so no cluster result is claimed.

