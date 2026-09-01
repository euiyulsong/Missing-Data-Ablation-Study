# missing_ablation.py

import argparse
import random
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

from ucimlrepo import fetch_ucirepo

import lightgbm as lgb


# ============================================================
# Seed
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. Load real missing-heavy dataset
#    UCI Horse Colic
# ============================================================

def load_data():

    print("=" * 100)
    print("Downloading UCI Horse Colic dataset")
    print("=" * 100)

    dataset = fetch_ucirepo(id=47)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    print("Raw X shape:", X.shape)
    print("Raw y shape:", y.shape)

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Outcome:
    # 1 = lived
    # 2 = died
    # 3 = euthanized
    #
    # 여기서는 binary:
    # lived = 0
    # died/euthanized = 1

    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    valid = y.notna()

    X = X.loc[valid].copy()
    y = y.loc[valid].copy()

    y = (
        y
        .map({
            1: 0,
            2: 1,
            3: 1,
        })
    )

    valid = y.notna()

    X = X.loc[valid].copy()
    y = y.loc[valid].astype(int)

    # 모든 feature를 numeric으로 변환
    # Horse Colic은 nominal code도 숫자로 되어 있어서
    # 이번 실험에서는 단순 tabular numerical input으로 사용
    for col in X.columns:
        X[col] = pd.to_numeric(
            X[col],
            errors="coerce",
        )

    df = X.copy()
    df["label"] = y.values

    print()
    print("Final rows:", len(df))
    print("Features:", X.shape[1])
    print("Positive rate:", df["label"].mean())

    return df


# ============================================================
# Missing summary
# ============================================================

def print_missing_summary(df):

    feature_cols = [
        c for c in df.columns
        if c != "label"
    ]

    missing = (
        df[feature_cols]
        .isna()
        .mean()
        .sort_values(ascending=False)
    )

    summary = pd.DataFrame({
        "feature": missing.index,
        "missing_rate": missing.values,
        "missing_pct": missing.values * 100,
    })

    print()
    print("=" * 100)
    print("MISSING VALUE SUMMARY")
    print("=" * 100)

    print(
        summary.to_string(
            index=False,
            formatters={
                "missing_rate":
                    lambda x: f"{x:.4f}",
                "missing_pct":
                    lambda x: f"{x:.2f}%"
            }
        )
    )

    return summary


# ============================================================
# Split
# ============================================================

def split_data(
    df,
    seed=42,
):

    train, temp = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=seed,
    )

    val, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["label"],
        random_state=seed,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ============================================================
# Drop columns based ONLY on TRAIN missing rate
# ============================================================

def select_columns_by_missing_rate(
    train,
    threshold,
):

    feature_cols = [
        c for c in train.columns
        if c != "label"
    ]

    if threshold is None:
        return feature_cols, []

    missing_rate = (
        train[feature_cols]
        .isna()
        .mean()
    )

    removed = (
        missing_rate[
            missing_rate > threshold
        ]
        .index
        .tolist()
    )

    selected = [
        c for c in feature_cols
        if c not in removed
    ]

    return selected, removed


# ============================================================
# Imputation
# ============================================================

def impute_data(
    train,
    val,
    test,
    feature_cols,
    method,
):

    train = train.copy()
    val = val.copy()
    test = test.copy()

    final_feature_cols = []

    for col in feature_cols:

        # -----------------------------------
        # Missing indicator
        # -----------------------------------

        if method == "median_flag":

            flag_col = (
                col
                + "__missing"
            )

            for df in [
                train,
                val,
                test,
            ]:
                df[flag_col] = (
                    df[col]
                    .isna()
                    .astype(np.float32)
                )

            final_feature_cols.append(
                flag_col
            )

        # -----------------------------------
        # Imputation
        # -----------------------------------

        if method in [
            "median",
            "median_flag",
        ]:

            fill_value = (
                train[col]
                .median()
            )

        elif method == "mean":

            fill_value = (
                train[col]
                .mean()
            )

        elif method == "zero":

            fill_value = 0.0

        else:
            raise ValueError(
                f"Unknown imputation: {method}"
            )

        if pd.isna(fill_value):
            fill_value = 0.0

        for df in [
            train,
            val,
            test,
        ]:

            df[col] = (
                df[col]
                .fillna(fill_value)
            )

        final_feature_cols.append(
            col
        )

    return (
        train,
        val,
        test,
        final_feature_cols,
    )


# ============================================================
# Standardization
# fit only on train
# ============================================================

def scale_data(
    train,
    val,
    test,
    feature_cols,
):

    train = train.copy()
    val = val.copy()
    test = test.copy()

    scaler = StandardScaler()

    train_values = scaler.fit_transform(
        train[feature_cols]
    )

    val_values = scaler.transform(
        val[feature_cols]
    )

    test_values = scaler.transform(
        test[feature_cols]
    )

    train.loc[
        :,
        feature_cols
    ] = train_values

    val.loc[
        :,
        feature_cols
    ] = val_values

    test.loc[
        :,
        feature_cols
    ] = test_values

    return (
        train,
        val,
        test,
    )


# ============================================================
# Dataset
# ============================================================

class TabularDataset(Dataset):

    def __init__(
        self,
        df,
        feature_cols,
    ):

        self.x = (
            df[feature_cols]
            .values
            .astype(np.float32)
        )

        self.y = (
            df["label"]
            .values
            .astype(np.float32)
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(
        self,
        idx,
    ):

        return (
            torch.tensor(
                self.x[idx],
                dtype=torch.float32,
            ),
            torch.tensor(
                self.y[idx],
                dtype=torch.float32,
            ),
        )


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(
        self,
        input_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.2
            ),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):

        return (
            self.net(x)
            .squeeze(-1)
        )


# ============================================================
# Evaluate MLP
# ============================================================

@torch.no_grad()
def evaluate_mlp(
    model,
    loader,
    device,
):

    model.eval()

    preds = []
    labels = []

    for x, y in loader:

        x = x.to(device)

        logits = model(x)

        probs = torch.sigmoid(
            logits
        )

        preds.extend(
            probs
            .cpu()
            .numpy()
        )

        labels.extend(
            y.numpy()
        )

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    auc = roc_auc_score(
        labels,
        preds,
    )

    ll = log_loss(
        labels,
        preds,
    )

    return auc, ll


# ============================================================
# Train MLP
# ============================================================

def train_mlp(
    train,
    val,
    test,
    feature_cols,
    args,
    device,
):

    train_ds = TabularDataset(
        train,
        feature_cols,
    )

    val_ds = TabularDataset(
        val,
        feature_cols,
    )

    test_ds = TabularDataset(
        test,
        feature_cols,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    model = MLP(
        len(feature_cols)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    best_auc = -1
    best_state = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0
        total_n = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(y)
            )

            total_n += len(y)

        val_auc, val_ll = (
            evaluate_mlp(
                model,
                val_loader,
                device,
            )
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss="
            f"{total_loss / total_n:.5f} "
            f"val_auc={val_auc:.5f} "
            f"val_logloss={val_ll:.5f}"
        )

        if val_auc > best_auc:

            best_auc = val_auc

            best_state = deepcopy(
                model.state_dict()
            )

    model.load_state_dict(
        best_state
    )

    test_auc, test_ll = (
        evaluate_mlp(
            model,
            test_loader,
            device,
        )
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    return {
        "test_auc": test_auc,
        "test_logloss": test_ll,
        "params": params,
    }


# ============================================================
# LightGBM
# Native missing handling
# ============================================================

def train_lightgbm(
    train,
    val,
    test,
    feature_cols,
    seed,
):

    X_train = train[
        feature_cols
    ]

    y_train = train[
        "label"
    ]

    X_val = val[
        feature_cols
    ]

    y_val = val[
        "label"
    ]

    X_test = test[
        feature_cols
    ]

    y_test = test[
        "label"
    ]

    model = lgb.LGBMClassifier(

        objective="binary",

        n_estimators=1000,

        learning_rate=0.03,

        num_leaves=15,

        max_depth=-1,

        min_child_samples=10,

        subsample=0.9,

        colsample_bytree=0.9,

        reg_lambda=1.0,

        random_state=seed,

        verbosity=-1,
    )

    model.fit(

        X_train,
        y_train,

        eval_set=[
            (
                X_val,
                y_val,
            )
        ],

        callbacks=[
            lgb.early_stopping(
                50,
                verbose=False,
            )
        ],
    )

    preds = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    auc = roc_auc_score(
        y_test,
        preds,
    )

    ll = log_loss(
        y_test,
        preds,
    )

    return {
        "test_auc": auc,
        "test_logloss": ll,
        "params":
            model.best_iteration_,
    }


# ============================================================
# MLP experiment
# ============================================================

def run_mlp_experiment(
    original_train,
    original_val,
    original_test,
    threshold,
    imputation,
    args,
    device,
):

    feature_cols, removed = (
        select_columns_by_missing_rate(
            original_train,
            threshold,
        )
    )

    (
        train,
        val,
        test,
        final_features,
    ) = impute_data(

        original_train[
            feature_cols
            + ["label"]
        ],

        original_val[
            feature_cols
            + ["label"]
        ],

        original_test[
            feature_cols
            + ["label"]
        ],

        feature_cols,

        imputation,
    )

    (
        train,
        val,
        test,
    ) = scale_data(
        train,
        val,
        test,
        final_features,
    )

    seed_everything(
        args.seed
    )

    result = train_mlp(
        train,
        val,
        test,
        final_features,
        args,
        device,
    )

    result.update({

        "model":
            "MLP",

        "drop_threshold":
            threshold,

        "imputation":
            imputation,

        "original_features":
            len(
                original_train.columns
            ) - 1,

        "selected_features":
            len(feature_cols),

        "final_features":
            len(final_features),

        "removed_columns":
            len(removed),

        "removed_names":
            ",".join(removed),
    })

    return result


# ============================================================
# LightGBM experiment
# ============================================================

def run_lightgbm_experiment(
    train,
    val,
    test,
    threshold,
    args,
):

    feature_cols, removed = (
        select_columns_by_missing_rate(
            train,
            threshold,
        )
    )

    # IMPORTANT:
    # No imputation
    # NaN remains NaN

    result = train_lightgbm(
        train,
        val,
        test,
        feature_cols,
        args.seed,
    )

    result.update({

        "model":
            "LightGBM",

        "drop_threshold":
            threshold,

        "imputation":
            "native_nan",

        "original_features":
            len(train.columns) - 1,

        "selected_features":
            len(feature_cols),

        "final_features":
            len(feature_cols),

        "removed_columns":
            len(removed),

        "removed_names":
            ",".join(removed),
    })

    return result


# ============================================================
# Main
# ============================================================

def main(args):

    seed_everything(
        args.seed
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    df = load_data()

    missing_summary = (
        print_missing_summary(
            df
        )
    )

    missing_summary.to_csv(
        "missing_summary.csv",
        index=False,
    )

    train, val, test = (
        split_data(
            df,
            args.seed,
        )
    )

    print()
    print(
        "Split sizes:"
    )

    print(
        "train:",
        len(train)
    )

    print(
        "val:",
        len(val)
    )

    print(
        "test:",
        len(test)
    )

    # --------------------------------------------------------
    # Experiment settings
    # --------------------------------------------------------

    thresholds = [

        (
            "none",
            None,
        ),

        (
            "drop_95",
            0.95,
        ),

        (
            "drop_90",
            0.90,
        ),

        (
            "drop_70",
            0.70,
        ),

        (
            "drop_50",
            0.50,
        ),

        (
            "drop_30",
            0.30,
        ),
    ]

    imputations = [
        "mean",
        "median",
        "zero",
        "median_flag",
    ]

    results = []

    # ========================================================
    # MLP
    # ========================================================

    for threshold_name, threshold in thresholds:

        for imputation in imputations:

            print()
            print(
                "=" * 100
            )

            print(
                "MLP EXPERIMENT"
            )

            print(
                "drop:",
                threshold_name,
                "| imputation:",
                imputation,
            )

            print(
                "=" * 100
            )

            result = run_mlp_experiment(
                train,
                val,
                test,
                threshold,
                imputation,
                args,
                device,
            )

            result[
                "variant"
            ] = (
                f"{threshold_name}"
                f"__"
                f"{imputation}"
            )

            results.append(
                result
            )

    # ========================================================
    # LightGBM
    # ========================================================

    for threshold_name, threshold in thresholds:

        print()
        print(
            "=" * 100
        )

        print(
            "LIGHTGBM EXPERIMENT"
        )

        print(
            "drop:",
            threshold_name,
            "| native NaN"
        )

        print(
            "=" * 100
        )

        result = (
            run_lightgbm_experiment(
                train,
                val,
                test,
                threshold,
                args,
            )
        )

        result[
            "variant"
        ] = (
            f"{threshold_name}"
            f"__native_nan"
        )

        results.append(
            result
        )

    # ========================================================
    # Results
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    result_cols = [

        "model",

        "variant",

        "drop_threshold",

        "imputation",

        "original_features",

        "selected_features",

        "final_features",

        "removed_columns",

        "test_auc",

        "test_logloss",

        "params",

        "removed_names",
    ]

    results_df = (
        results_df[
            result_cols
        ]
    )

    print()
    print(
        "=" * 140
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 140
    )

    print(
        results_df
        .sort_values(
            "test_auc",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    results_df.to_csv(
        "missing_ablation_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Best MLP
    # --------------------------------------------------------

    mlp_df = (
        results_df[
            results_df[
                "model"
            ] == "MLP"
        ]
    )

    best_mlp = (
        mlp_df
        .sort_values(
            "test_auc",
            ascending=False,
        )
        .iloc[0]
    )

    # --------------------------------------------------------
    # Best LightGBM
    # --------------------------------------------------------

    lgb_df = (
        results_df[
            results_df[
                "model"
            ] == "LightGBM"
        ]
    )

    best_lgb = (
        lgb_df
        .sort_values(
            "test_auc",
            ascending=False,
        )
        .iloc[0]
    )

    print()
    print(
        "=" * 100
    )

    print(
        "BEST MLP"
    )

    print(
        "=" * 100
    )

    print(
        best_mlp.to_string()
    )

    print()
    print(
        "=" * 100
    )

    print(
        "BEST LIGHTGBM"
    )

    print(
        "=" * 100
    )

    print(
        best_lgb.to_string()
    )

    print()
    print(
        "Saved:"
    )

    print(
        "  missing_summary.csv"
    )

    print(
        "  missing_ablation_results.csv"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(args)
