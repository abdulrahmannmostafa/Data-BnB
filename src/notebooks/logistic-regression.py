#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression — Demand Classification
# 
# **Target:** `demand_label` (binary: 0 = low demand, 1 = high demand)  
# **Tuning:** Optuna (300 trials, weighted-F1 objective)  
# **Note:** `Price_vs_city_median` and `Parsed Amenities` are excluded (leakage / meaningless encoding).

# In[20]:


get_ipython().system('pip install mlflow')


# In[21]:


get_ipython().system('pip install pyngrok')


# In[22]:


get_ipython().system('ngrok config add-authtoken 3D8Z9KARgl0UCcT6wcurJSjWDrz_4casNGXJeX1aqLjRAS3oR')


# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score,
)
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.style.use('dark_background')
SEED = 42


# %%

# In[30]:


import subprocess, time
from pyngrok import ngrok
import mlflow

# 1. Get or create ngrok tunnel first
subprocess.run(["fuser", "-k", "5000/tcp"], capture_output=True)

tunnels = ngrok.get_tunnels()
if tunnels:
    public_url = tunnels[0].public_url
else:
    public_url = ngrok.connect(addr="127.0.0.1:5000")

ngrok_host = str(public_url).replace("https://", "").replace("http://", "")
print("Ngrok host:", ngrok_host)

# 2. Start MLflow knowing the hostname
subprocess.Popen([
    "mlflow", "server",
    "--host", "0.0.0.0",
    "--port", "5000",
    "--backend-store-uri", "sqlite:////kaggle/working/mlflow.db",
    "--default-artifact-root", "/kaggle/working/mlflow_artifacts",
    "--allowed-hosts", f"{ngrok_host},localhost,127.0.0.1"
])
time.sleep(8)

# 3. Connect client
mlflow.set_tracking_uri(str(public_url))
mlflow.set_experiment("Big-Data-Project-Team-1")
print("Ready!", public_url)


# ## 1. Load Data

# In[ ]:


train_df = pd.read_csv('/kaggle/input/datasets/abdulrahmanmostafa10/data-bnb/train.csv', low_memory=False)
test_df  = pd.read_csv('/kaggle/input/datasets/abdulrahmanmostafa10/data-bnb/test.csv',  low_memory=False)

TARGET = 'demand_label'   # 0 = low demand, 1 = high demand

# Columns to exclude from features
DROP_COLS = [
    'demand_label', 'demand_label_3', 'demand_score',
    'Price_log', 'Price_original',
    'Price_vs_city_median',   # removed: derived from price target -> leakage
]
raw_cols     = [c for c in train_df.columns if c.endswith('_raw')]
amenity_cols = [c for c in train_df.columns if 'Parsed Amenities' in c]
DROP_COLS   += raw_cols + amenity_cols

feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

X_all  = train_df[feature_cols].fillna(0)
y_all  = train_df[TARGET]
X_test = test_df[feature_cols].fillna(0)
y_test = test_df[TARGET]

# Validation split from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all
)

print(f'Train : {X_train.shape} | Val : {X_val.shape} | Test : {X_test.shape}')
print(f'Class balance (train): {y_train.value_counts(normalize=True).round(3).to_dict()}')


# ## 2. Hyperparameter Tuning (Optuna)

# In[ ]:


def objective(trial):
    solver  = trial.suggest_categorical('solver',  ['liblinear', 'saga', 'lbfgs'])
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])

    # Prune invalid solver/penalty combos
    if solver == 'lbfgs'     and penalty != 'l2':         raise optuna.exceptions.TrialPruned()
    if solver == 'liblinear' and penalty not in ['l1','l2']: raise optuna.exceptions.TrialPruned()
    if solver != 'saga'      and penalty == 'elasticnet': raise optuna.exceptions.TrialPruned()

    params = {
        'C':            trial.suggest_float('C', 1e-4, 100.0, log=True),
        'solver':       solver,
        'penalty':      penalty,
        'max_iter':    10000,
        'random_state': SEED,
        'class_weight': 'balanced',
    }
    if solver == 'liblinear':
        params['intercept_scaling'] = trial.suggest_float('intercept_scaling', 0.1, 5.0)
    if penalty == 'elasticnet':
        params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

    model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(**params))
])
    model.fit(X_train, y_train)
    return f1_score(y_val, model.predict(X_val), average='weighted')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=2)
print(f'Best Val F1 : {study.best_value:.4f}')
print(f'Best Params : {study.best_params}')


# ## 3. Train Final Model on Train + Val

# In[ ]:


best_params = study.best_params
solver  = best_params.get('solver',  'lbfgs')
penalty = best_params.get('penalty', 'l2')

# Filter conditional params to avoid TypeError
final_params = {
    k: v for k, v in best_params.items()
    if not (k == 'l1_ratio'           and penalty != 'elasticnet')
    and not (k == 'intercept_scaling' and solver  != 'liblinear')
}

X_combined = pd.concat([X_train, X_val])
y_combined = pd.concat([y_train, y_val])

model = LogisticRegression(**final_params, max_iter=2000, random_state=SEED, class_weight='balanced')
model.fit(X_combined, y_combined)
print('Final model trained on', len(X_combined), 'samples.')


# ## 4. Evaluation on Hold-Out Test Set

# In[ ]:


y_pred = model.predict(X_test)

print(f'Test Accuracy : {accuracy_score(y_test, y_pred):.4f}')
print(f'F1 (weighted) : {f1_score(y_test, y_pred, average="weighted"):.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['Low Demand', 'High Demand']))


# In[ ]:


fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=['Low Demand', 'High Demand']
).plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix — Logistic Regression', fontsize=14)
plt.tight_layout()
plt.show()


# ## 5. Learning Curve

# In[ ]:


lc_model = LogisticRegression(**final_params, max_iter=2000, random_state=SEED, class_weight='balanced')

train_sizes, train_scores, val_scores = learning_curve(
    lc_model, X_train, y_train,
    cv=5, scoring='f1_weighted',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_scores.mean(axis=1), marker='o', label='Train F1')
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2)
ax.plot(train_sizes, val_scores.mean(axis=1), marker='s', label='Val F1')
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2)
ax.set_xlabel('Training Size')
ax.set_ylabel('F1 Score (Weighted)')
ax.set_title('Learning Curve — Logistic Regression')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




