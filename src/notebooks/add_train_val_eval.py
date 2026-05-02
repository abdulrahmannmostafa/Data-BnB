#!/usr/bin/env python3
"""
Script to add train/validation evaluation cells to all model notebooks.
For each model, inserts a new cell right before the test evaluation cell
that prints accuracy, F1 score, and classification report on train and val sets.
"""

import json
import copy
import uuid

def make_code_cell(source_lines, cell_id=None):
    """Create a notebook code cell dict."""
    if cell_id is None:
        cell_id = str(uuid.uuid4())[:8]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"train-val-eval-{cell_id}",
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def make_markdown_cell(source_lines, cell_id=None):
    """Create a notebook markdown cell dict."""
    if cell_id is None:
        cell_id = str(uuid.uuid4())[:8]
    return {
        "cell_type": "markdown",
        "id": f"md-train-val-eval-{cell_id}",
        "metadata": {},
        "source": source_lines
    }

def build_train_val_eval_source(model_name, model_var, target_names_str):
    """Build source lines for train+val evaluation cell."""
    lines = [
        f"# --- Results on Train Set ---\n",
        f"y_pred_train = {model_var}.predict(X_train)\n",
        f"\n",
        f"print('=== {model_name} — Train Set ===')\n",
        f"print(f'Accuracy      : {{accuracy_score(y_train, y_pred_train):.4f}}')\n",
        f"print(f'F1 (weighted) : {{f1_score(y_train, y_pred_train, average=\"weighted\"):.4f}}')\n",
        f"print(classification_report(y_train, y_pred_train, target_names={target_names_str}))\n",
        f"\n",
        f"# --- Results on Validation Set ---\n",
        f"y_pred_val = {model_var}.predict(X_val)\n",
        f"\n",
        f"print('=== {model_name} — Validation Set ===')\n",
        f"print(f'Accuracy      : {{accuracy_score(y_val, y_pred_val):.4f}}')\n",
        f"print(f'F1 (weighted) : {{f1_score(y_val, y_pred_val, average=\"weighted\"):.4f}}')\n",
        f"print(classification_report(y_val, y_pred_val, target_names={target_names_str}))\n",
    ]
    return lines

def find_cell_index_containing(cells, search_str):
    """Find index of cell whose source contains search_str."""
    for i, cell in enumerate(cells):
        source = cell.get("source", [])
        if isinstance(source, list):
            full = "".join(source)
        else:
            full = source
        if search_str in full:
            return i
    return -1

def process_notebook(filepath, insertions):
    """
    Process a notebook file and insert train/val eval cells.
    
    insertions: list of dicts with keys:
        - search_str: string to find in existing cell (to locate test eval cell)
        - model_name: human readable model name
        - model_var: Python variable name for the model
        - target_names_str: Python repr of target_names list
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb["cells"]
    
    # Process insertions in reverse order (so indices don't shift)
    offset = 0
    insert_ops = []
    for ins in insertions:
        idx = find_cell_index_containing(cells, ins["search_str"])
        if idx == -1:
            print(f"  WARNING: Could not find cell containing '{ins['search_str']}' in {filepath}")
            continue
        insert_ops.append((idx, ins))
    
    # Sort by index descending so insertions don't affect each other
    insert_ops.sort(key=lambda x: x[0], reverse=True)
    
    for idx, ins in insert_ops:
        # Build the markdown header cell
        md_cell = make_markdown_cell(
            [f"### {ins['model_name']} — Train & Validation Results"],
            cell_id=ins["model_var"]
        )
        # Build the code cell
        source = build_train_val_eval_source(
            ins["model_name"], ins["model_var"], ins["target_names_str"]
        )
        code_cell = make_code_cell(source, cell_id=ins["model_var"])
        
        # Insert before the test eval cell
        cells.insert(idx, code_cell)
        cells.insert(idx, md_cell)
        print(f"  Inserted train/val eval for {ins['model_name']} at index {idx}")
    
    nb["cells"] = cells
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"  Saved: {filepath}")

def main():
    base = "/home/amostafa/CMP/Data-BnB/src/notebooks"
    
    # 1. logistic-regression.ipynb
    print("Processing logistic-regression.ipynb...")
    process_notebook(f"{base}/logistic-regression.ipynb", [
        {
            "search_str": "model.predict(X_test)",
            "model_name": "Logistic Regression",
            "model_var": "model",
            "target_names_str": "['Low Demand', 'High Demand']"
        }
    ])
    
    # 2. naive-bayes-mlp.ipynb
    print("Processing naive-bayes-mlp.ipynb...")
    process_notebook(f"{base}/naive-bayes-mlp.ipynb", [
        {
            "search_str": "nb_model.predict(X_test)",
            "model_name": "Gaussian Naive Bayes",
            "model_var": "nb_model",
            "target_names_str": "['Low','High']"
        },
        {
            "search_str": "mlp_model.predict(X_test)",
            "model_name": "MLP",
            "model_var": "mlp_model",
            "target_names_str": "['Low','High']"
        }
    ])
    
    # 3. svm-knn.ipynb
    print("Processing svm-knn.ipynb...")
    process_notebook(f"{base}/svm-knn.ipynb", [
        {
            "search_str": "svm_model.predict(X_test)",
            "model_name": "SVM",
            "model_var": "svm_model",
            "target_names_str": "['Low','High']"
        },
        {
            "search_str": "knn_model.predict(X_test)",
            "model_name": "KNN",
            "model_var": "knn_model",
            "target_names_str": "['Low','High']"
        }
    ])
    
    # 4. tree-models.ipynb
    print("Processing tree-models.ipynb...")
    process_notebook(f"{base}/tree-models.ipynb", [
        {
            "search_str": "rf_model.predict(X_test)",
            "model_name": "Random Forest",
            "model_var": "rf_model",
            "target_names_str": "['Low Demand','High Demand']"
        },
        {
            "search_str": "et_model.predict(X_test)",
            "model_name": "Extra Trees",
            "model_var": "et_model",
            "target_names_str": "['Low Demand','High Demand']"
        }
    ])
    
    # 5. boosting-models.ipynb
    print("Processing boosting-models.ipynb...")
    process_notebook(f"{base}/boosting-models.ipynb", [
        {
            "search_str": "xgb_model.predict(X_test)",
            "model_name": "XGBoost",
            "model_var": "xgb_model",
            "target_names_str": "['Low','High']"
        },
        {
            "search_str": "lgbm_model.predict(X_test)",
            "model_name": "LightGBM",
            "model_var": "lgbm_model",
            "target_names_str": "['Low','High']"
        },
        {
            "search_str": "cat_model.predict(X_test)",
            "model_name": "CatBoost",
            "model_var": "cat_model",
            "target_names_str": "['Low','High']"
        }
    ])
    
    print("\nDone! All notebooks updated.")

if __name__ == "__main__":
    main()
