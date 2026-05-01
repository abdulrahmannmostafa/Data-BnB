import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import optuna

plt.style.use("dark_background")


# ══════════════════════════════════════════════════════════════════
# Logistic Regression
# ══════════════════════════════════════════════════════════════════
class LogisticRegressionModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_logreg_model = None
        self.best_logreg_params = {}

    def objective_logreg(self, trial):
        solver = trial.suggest_categorical("solver", ["liblinear", "saga", "lbfgs"])
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

        # Prune invalid solver / penalty combos
        if solver == "lbfgs" and penalty != "l2":
            raise optuna.exceptions.TrialPruned()
        if solver == "liblinear" and penalty not in ["l1", "l2"]:
            raise optuna.exceptions.TrialPruned()
        if solver != "saga" and penalty == "elasticnet":
            raise optuna.exceptions.TrialPruned()

        param = {
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "solver": solver,
            "penalty": penalty,
            "max_iter": 2000,
            "random_state": 42,
            "class_weight": "balanced",
            # FIX: multi_class removed — deprecated in modern sklearn
        }

        if solver == "liblinear":
            param["intercept_scaling"] = trial.suggest_float(
                "intercept_scaling", 0.1, 5.0
            )

        if penalty == "elasticnet":
            param["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        model = LogisticRegression(**param)
        model.fit(self.X_train, self.y_train)
        y_val_pred = model.predict(self.X_val)
        return f1_score(self.y_val, y_val_pred, average="weighted")

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_logreg, n_trials=300, n_jobs=2)

        self.best_logreg_params = study.best_params
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        # FIX 2: filter out conditional params that only apply to specific
        # solver/penalty combos — prevents TypeError when reusing best_params
        solver  = self.best_logreg_params.get("solver", "lbfgs")
        penalty = self.best_logreg_params.get("penalty", "l2")
        final_params = {
            k: v for k, v in self.best_logreg_params.items()
            if not (k == "l1_ratio" and penalty != "elasticnet")
            and not (k == "intercept_scaling" and solver != "liblinear")
        }

        self.best_logreg_model = LogisticRegression(
            **final_params,
            max_iter=2000,
            random_state=42,
        )
        self.best_logreg_model.fit(X_combined, y_combined)

    def model_predict(self):
        y_pred = self.best_logreg_model.predict(self.X_test)
        print(f"\nTest Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        print(
            f"F1 (weighted) : {f1_score(self.y_test, y_pred, average='weighted'):.4f}"
        )
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def plot_learning_curve(self):
        model = LogisticRegression(
            **self.best_logreg_params, max_iter=2000, random_state=42
        )
        self._plot_curve(model, "Logistic Regression")

    # ── shared helper ──────────────────────────────────────────────
    def _plot_curve(self, model, title):
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
        plt.xlabel("Training Size")
        plt.ylabel("F1 Score (Weighted)")
        plt.title(f"Learning Curve — {title}")
        plt.legend()
        plt.grid()
        plt.show()


# ══════════════════════════════════════════════════════════════════
# KNN
# ══════════════════════════════════════════════════════════════════
class KNNModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_knn_model = None
        self.best_knn_params = {}

    def objective_knn(self, trial):
        param = {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical(
                "metric", ["euclidean", "manhattan", "minkowski"]
            ),
        }
        model = KNeighborsClassifier(**param)
        model.fit(self.X_train, self.y_train)
        return f1_score(self.y_val, model.predict(self.X_val), average="weighted")

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_knn, n_trials=300, n_jobs=2)

        self.best_knn_params = study.best_params
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_knn_model = KNeighborsClassifier(**self.best_knn_params)
        self.best_knn_model.fit(X_combined, y_combined)

    def model_predict(self):
        y_pred = self.best_knn_model.predict(self.X_test)
        print(f"\nTest Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        print(
            f"F1 (weighted) : {f1_score(self.y_test, y_pred, average='weighted'):.4f}"
        )
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def plot_learning_curve(self):
        model = KNeighborsClassifier(**self.best_knn_params)
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
        plt.xlabel("Training Size")
        plt.ylabel("F1 Score (Weighted)")
        plt.title("Learning Curve — KNN")
        plt.legend()
        plt.grid()
        plt.show()


# ══════════════════════════════════════════════════════════════════
# Random Forest
# ══════════════════════════════════════════════════════════════════
class RFModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_rf_model = None
        self.best_rf_params = {}

    def objective_rf(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        model = RandomForestClassifier(
            **param, random_state=42, class_weight="balanced"
        )
        model.fit(self.X_train, self.y_train)
        return f1_score(self.y_val, model.predict(self.X_val), average="weighted")

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_rf, n_trials=100, n_jobs=2)

        self.best_rf_params = study.best_params
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_rf_model = RandomForestClassifier(
            **self.best_rf_params, random_state=42, class_weight="balanced"
        )
        self.best_rf_model.fit(X_combined, y_combined)

    def model_predict(self):
        y_pred = self.best_rf_model.predict(self.X_test)
        print(f"\nTest Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        print(
            f"F1 (weighted) : {f1_score(self.y_test, y_pred, average='weighted'):.4f}"
        )
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def plot_learning_curve(self):
        model = RandomForestClassifier(
            **self.best_rf_params, random_state=42, class_weight="balanced"
        )
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
        plt.xlabel("Training Size")
        plt.ylabel("F1 Score (Weighted)")
        plt.title("Learning Curve — Random Forest")
        plt.legend()
        plt.grid()
        plt.show()


# ══════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════
class MLPModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_mlp_model = None
        self.best_mlp_params = {}

    def objective_mlp(self, trial):
        # FIX: all tunable params are now actually USED in the model below
        param = {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", [(64,), (128,), (64, 32), (128, 64), (256, 128)]
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-1, log=True
            ),
        }

        model = MLPClassifier(
            **param,  # ← FIX: use trial params, not hardcoded values
            solver="adam",
            max_iter=500,
            shuffle=True,
            random_state=42,
        )
        model.fit(self.X_train, self.y_train)
        return f1_score(self.y_val, model.predict(self.X_val), average="weighted")

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_mlp, n_trials=50, n_jobs=2)

        self.best_mlp_params = study.best_params
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_mlp_model = MLPClassifier(
            **self.best_mlp_params,
            solver="adam",
            max_iter=500,
            shuffle=True,
            random_state=42,
        )
        self.best_mlp_model.fit(X_combined, y_combined)

    def model_predict(self):
        y_pred = self.best_mlp_model.predict(self.X_test)
        print(f"\nTest Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        print(
            f"F1 (weighted) : {f1_score(self.y_test, y_pred, average='weighted'):.4f}"
        )
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def plot_learning_curve(self):
        model = MLPClassifier(
            **self.best_mlp_params, solver="adam", max_iter=500, random_state=42
        )
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
        plt.xlabel("Training Size")
        plt.ylabel("F1 Score (Weighted)")
        plt.title("Learning Curve — MLP")
        plt.legend()
        plt.grid()
        plt.show()


# ══════════════════════════════════════════════════════════════════
# SVM
# ══════════════════════════════════════════════════════════════════
class SVMModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_svm_model = None
        self.best_svm_params = {}

    def objective_svm(self, trial):
        param = {
            "C": trial.suggest_float("C", 1e-3, 100, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
        model = SVC(**param, class_weight="balanced")
        model.fit(self.X_train, self.y_train)
        return f1_score(self.y_val, model.predict(self.X_val), average="weighted")

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_svm, n_trials=50, n_jobs=2)

        self.best_svm_params = study.best_params
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_svm_model = SVC(**self.best_svm_params, class_weight="balanced")
        self.best_svm_model.fit(X_combined, y_combined)

    def model_predict(self):
        y_pred = self.best_svm_model.predict(self.X_test)
        print(f"\nTest Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        print(
            f"F1 (weighted) : {f1_score(self.y_test, y_pred, average='weighted'):.4f}"
        )
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def plot_learning_curve(self):
        model = SVC(**self.best_svm_params, class_weight="balanced")
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
        plt.xlabel("Training Size")
        plt.ylabel("F1 Score (Weighted)")
        plt.title("Learning Curve — SVM")
        plt.legend()
        plt.grid()
        plt.show()


# ══════════════════════════════════════════════════════════════════
# Naive Bayes
# ══════════════════════════════════════════════════════════════════
class NaiveBayesModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_nb_model = None
        self.best_nb_params = {}

    def objective_nb(self, trial):
        var_smoothing = trial.suggest_float("var_smoothing", 1e-11, 1e-7, log=True)
        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(self.X_train, self.y_train)
        # FIX: use f1_weighted (not accuracy) to be consistent with all other models
        return f1_score(self.y_val, model.predict(self.X_val), average="weighted")

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_nb, n_trials=200, n_jobs=2)

        self.best_nb_params = study.best_params
        print(f"Best GaussianNB params: {self.best_nb_params}")

        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.best_nb_model = GaussianNB(**self.best_nb_params)
        self.best_nb_model.fit(X_combined, y_combined)

    def model_predict(self):
        y_pred = self.best_nb_model.predict(self.X_test)
        print(f"\nTest Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        print(
            f"F1 (weighted) : {f1_score(self.y_test, y_pred, average='weighted'):.4f}"
        )
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    # FIX 3: NaiveBayesModel was missing plot_learning_curve (all other models have it)
    def plot_learning_curve(self):
        model = GaussianNB(**self.best_nb_params)
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
        plt.xlabel("Training Size")
        plt.ylabel("F1 Score (Weighted)")
        plt.title("Learning Curve — Naive Bayes")
        plt.legend()
        plt.grid()
        plt.show()


# ══════════════════════════════════════════════════════════════════
# XGBoost
# ══════════════════════════════════════════════════════════════════
class XGBModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_xgb_model = None
        self.best_xgb_params = {}

        n_classes = y_train.nunique()
        # FIX: scale_pos_weight is binary-only — only set for 2-class problems
        if n_classes == 2:
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            self.scale_pos_weight = neg / pos if pos > 0 else 1.0
        else:
            self.scale_pos_weight = None  # use sample_weight for multi-class instead

        self.n_classes = n_classes

    def objective_xgb(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "eval_metric": "mlogloss",
            # FIX 1: use_label_encoder removed — deprecated & removed in XGBoost ≥ 2.x
        }
        if self.scale_pos_weight is not None:
            param["scale_pos_weight"] = self.scale_pos_weight

        model = XGBClassifier(**param)
        model.fit(self.X_train, self.y_train)
        return f1_score(self.y_val, model.predict(self.X_val), average="weighted")

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_xgb, n_trials=150, n_jobs=2)

        self.best_xgb_params = study.best_params
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        final_params = {
            **self.best_xgb_params,
            "eval_metric": "mlogloss",
            # FIX 1: use_label_encoder removed — deprecated & removed in XGBoost ≥ 2.x
        }
        if self.scale_pos_weight is not None:
            final_params["scale_pos_weight"] = self.scale_pos_weight

        self.best_xgb_model = XGBClassifier(**final_params)
        self.best_xgb_model.fit(X_combined, y_combined)

    def model_predict(self):
        y_pred = self.best_xgb_model.predict(self.X_test)
        print(f"\nTest Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        print(
            f"F1 (weighted) : {f1_score(self.y_test, y_pred, average='weighted'):.4f}"
        )
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def plot_learning_curve(self):
        params = {
            **self.best_xgb_params,
            "eval_metric": "mlogloss",
            # FIX 1: use_label_encoder removed
        }
        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight
        model = XGBClassifier(**params)
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
        plt.xlabel("Training Size")
        plt.ylabel("F1 Score (Weighted)")
        plt.title("Learning Curve — XGBoost")
        plt.legend()
        plt.grid()
        plt.show()


# ══════════════════════════════════════════════════════════════════
# Extra Trees
# ══════════════════════════════════════════════════════════════════
class ExtraTreesModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.best_params = {}

    def objective_et(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        model = ExtraTreesClassifier(**param, random_state=42, class_weight="balanced")
        model.fit(self.X_train, self.y_train)
        return f1_score(self.y_val, model.predict(self.X_val), average="weighted")

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_et, n_trials=50, n_jobs=2)

        self.best_params = study.best_params
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.model = ExtraTreesClassifier(
            **self.best_params, random_state=42, class_weight="balanced"
        )
        self.model.fit(X_combined, y_combined)

    def model_predict(self):
        y_pred = self.model.predict(self.X_test)
        print(f"\nTest Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        print(
            f"F1 (weighted) : {f1_score(self.y_test, y_pred, average='weighted'):.4f}"
        )
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def plot_learning_curve(self):
        model = ExtraTreesClassifier(
            **self.best_params, random_state=42, class_weight="balanced"
        )
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
        plt.xlabel("Training Size")
        plt.ylabel("F1 Score (Weighted)")
        plt.title("Learning Curve — Extra Trees")
        plt.legend()
        plt.grid()
        plt.show()


# ══════════════════════════════════════════════════════════════════
# CatBoost
# ══════════════════════════════════════════════════════════════════
class CatBoostModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = None
        self.best_params = {}

    def objective_cat(self, trial):
        param = {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "verbose": 0,
        }
        model = CatBoostClassifier(**param)
        model.fit(self.X_train, self.y_train)
        return f1_score(self.y_val, model.predict(self.X_val), average="weighted")

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_cat, n_trials=100, n_jobs=2)

        self.best_params = study.best_params
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        self.model = CatBoostClassifier(**self.best_params, verbose=0)
        self.model.fit(X_combined, y_combined)

    def model_predict(self):
        y_pred = self.model.predict(self.X_test)
        print(f"\nTest Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        print(
            f"F1 (weighted) : {f1_score(self.y_test, y_pred, average='weighted'):.4f}"
        )
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def plot_learning_curve(self):
        model = CatBoostClassifier(**self.best_params, verbose=0)
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
        plt.xlabel("Training Size")
        plt.ylabel("F1 Score (Weighted)")
        plt.title("Learning Curve — CatBoost")
        plt.legend()
        plt.grid()
        plt.show()


# ══════════════════════════════════════════════════════════════════
# LightGBM
# ══════════════════════════════════════════════════════════════════
class LGBMModel:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.best_lgbm_model = None
        self.best_lgbm_params = {}

        n_classes = y_train.nunique()
        # FIX: scale_pos_weight binary-only guard (same as XGB)
        if n_classes == 2:
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            self.scale_pos_weight = neg / pos if pos > 0 else 1.0
        else:
            self.scale_pos_weight = None

    def objective_lgbm(self, trial):
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", -1, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        if self.scale_pos_weight is not None:
            param["scale_pos_weight"] = self.scale_pos_weight

        model = LGBMClassifier(**param, verbose=-1)
        model.fit(self.X_train, self.y_train)
        return f1_score(self.y_val, model.predict(self.X_val), average="weighted")

    def model_train(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective_lgbm, n_trials=150, n_jobs=2)

        self.best_lgbm_params = study.best_params
        X_combined = pd.concat([self.X_train, self.X_val])
        y_combined = pd.concat([self.y_train, self.y_val])

        final_params = {**self.best_lgbm_params}
        if self.scale_pos_weight is not None:
            final_params["scale_pos_weight"] = self.scale_pos_weight

        self.best_lgbm_model = LGBMClassifier(**final_params, verbose=-1)
        self.best_lgbm_model.fit(X_combined, y_combined)

    def model_predict(self):
        y_pred = self.best_lgbm_model.predict(self.X_test)
        print(f"\nTest Accuracy : {accuracy_score(self.y_test, y_pred):.4f}")
        print(
            f"F1 (weighted) : {f1_score(self.y_test, y_pred, average='weighted'):.4f}"
        )
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def plot_learning_curve(self):
        params = {**self.best_lgbm_params}
        if self.scale_pos_weight is not None:
            params["scale_pos_weight"] = self.scale_pos_weight
        model = LGBMClassifier(**params, verbose=-1)
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            self.X_train,
            self.y_train,
            cv=5,
            scoring="f1_weighted",
            train_sizes=np.linspace(0.1, 1.0, 10),
            n_jobs=-1,
        )
        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_scores.mean(axis=1), label="Train F1")
        plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation F1")
        plt.xlabel("Training Size")
        plt.ylabel("F1 Score (Weighted)")
        plt.title("Learning Curve — LightGBM")
        plt.legend()
        plt.grid()
        plt.show()
