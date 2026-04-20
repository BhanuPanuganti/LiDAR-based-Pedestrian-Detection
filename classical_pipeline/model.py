import xgboost as xgb
import torch
import os

def get_model(spw):
    return xgb.XGBClassifier(
        # Core
        n_estimators=500,
        max_depth=8,
        learning_rate=0.03,

        # Regularization (VERY IMPORTANT for your case)
        gamma=1.5,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,

        # Sampling (reduces overfitting + speeds up GPU)
        subsample=0.8,
        colsample_bytree=0.7,

        # Imbalance handling (CAP IT)
        scale_pos_weight=min(spw, 120),

        # GPU acceleration
        tree_method='hist',
        device='cuda',

        # Misc
        eval_metric='logloss',
        max_bin=256,
        random_state=42,
        n_jobs=-1
    )

def train_or_load(model, X_tr, y_tr, X_te, y_te, path):

    if os.path.exists(path):
        print("[INFO] Loading model...")
        model.load_model(path)
    else:
        print("[INFO] Training model...")
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50)
        model.save_model(path)

    return model