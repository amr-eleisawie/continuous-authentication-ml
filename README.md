# Continuous Authentication ML

Can a system keep checking whether a session still belongs to its authorized user after login? This project explores that question with a behavior-focused machine-learning risk model and two small demo interfaces.

> **Research prototype:** predictions are demonstrations, not a production security control. Do not use them to grant or deny real access.

## What it does

The analysis classifies each session as **Authorized** or **Unauthorized** using behavioral and system measurements. It compares a majority-class baseline, Logistic Regression, Random Forest, and Gradient Boosting. The training workflow removes possible post-decision leakage fields and keeps participant IDs out of the model features.

The data is split by participant with `GroupShuffleSplit`, then models are compared with `GroupKFold`. This helps test generalization to participants not seen during training instead of letting sessions from the same person appear on both sides of a random row split.

The selected behavior-focused Random Forest excludes `Active_Application` and `Command_Type`. The project report records **86.65% test accuracy**, **85.11% test F1**, and **94.59% test ROC-AUC** for this tuned model. A full-feature comparison scores higher, but depends heavily on application and command categories. Because those relationships may be synthetic or rule-generated, the scores should not be interpreted as expected real-world performance.

## Repository layout

```text
app.py                                           Streamlit demo
notebooks/continuous_authentication_analysis.ipynb  Data analysis and model evaluation
outputs/                                         Detailed project explanations
tuned_behavior_focused_pipeline.pkl              Model used by the Streamlit app
vercel_app/                                      Static demo and prediction endpoint
requirements.txt                                 Local Streamlit app dependencies
```

The Vercel demo is available at [continuous-authentication-ml.vercel.app](https://continuous-authentication-ml.vercel.app). The Streamlit app returns one of three risk-based actions: `ALLOW`, `REQUEST_OTP`, or `BLOCK_AND_ALERT`.

## Run locally

Use Python and run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

The demo uses the checked-in model artifact and does not need the training dataset. To run the analysis notebook, obtain `continuous_auth_dataset.csv` separately and place it in the repository root. The dataset is intentionally not committed; the ignore rules help prevent accidental publication of session-level behavioral data.

Open `notebooks/continuous_authentication_analysis.ipynb` and run its cells in order. Its saved outputs were cleared for publication; results are regenerated when the notebook is run with the dataset.

## Deployment and integrations

- `vercel_app/` contains the static interface and `/api/predict` endpoint.
- The demo can send prediction results to an n8n webhook when `N8N_WEBHOOK_URL` is configured. Keep webhook URLs and other secrets in environment variables, never in source control.
- The model is serialized with joblib. Load model files only from a trusted source and use compatible Python/scikit-learn versions.

## Limitations

- The dataset contains 5,000 sessions from 400 participants and appears synthetic or rule-generated; performance on real users may be substantially different.
- Application and command categories carry unusually strong class signal. The reduced model excludes them, but still requires evaluation on representative, independently collected data.
- `Previous_Authentication_Score` must be available before the prediction being made; otherwise it could introduce target leakage in a real deployment.
- The app is an educational prototype and does not implement production-grade identity verification, privacy controls, monitoring, or security review.

See [`outputs/Project_A_to_Z_Explanation.md`](outputs/Project_A_to_Z_Explanation.md) for the full methodology and [`outputs/Person_3_Machine_Learning_Explanation.md`](outputs/Person_3_Machine_Learning_Explanation.md) for the modeling details.
