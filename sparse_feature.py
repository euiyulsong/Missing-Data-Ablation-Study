# titanic_sparse_feature_ablation.py

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier, early_stopping

try:
    import seaborn as sns
except ImportError:
    raise ImportError(
        "Install dependencies first:\n"
        "pip install seaborn pandas numpy scikit-learn lightgbm"
    )


SEED = 42


def load_data():
    # seaborn의 공개 Titanic dataset 자동 다운로드
    df = sns.load_dataset("titanic")

    # target
    df = df.dropna(subset=["survived"]).copy()
    df["survived"] = df["survived"].astype(int)

    # 원본 Cabin 컬럼이 seaborn titanic에는 없음.
    # 그래서 deck을 sparse cabin-derived feature로 사용.
    #
    # deck은 Cabin 첫 문자에서 파생된 정보에 해당하고
    # missing 비율이 높음.
    return df


def print_missing_summary(df):
    print("\n" + "=" * 100)
    print("MISSING SUMMARY")
    print("=" * 100)

    s = df.isna().mean().sort_values(ascending=False)
    out = pd.DataFrame({
        "feature": s.index,
        "missing_rate": s.values,
        "missing_pct": s.values * 100,
    })

    print(out.to_string(index=False))
    print()


def add_sparse_features(df):
    df = df.copy()

    # deck 자체가 Cabin-derived sparse categorical feature
    df["deck_raw"] = df["deck"].astype(object)

    # Cabin/deck 존재 여부
    df["deck_known"] = df["deck"].notna().astype(int)

    # missing 자체를 category로
    df["deck_missing_category"] = (
        df["deck"]
        .astype(object)
        .where(df["deck"].notna(), "__MISSING__")
        .astype(str)
    )

    return df


BASE_NUMERIC = [
    "age",
    "sibsp",
    "parch",
    "fare",
]

BASE_CATEGORICAL = [
    "sex",
    "class",
    "embarked",
    "who",
    "alone",
]


VARIANTS = {
    # sparse feature 완전 제거
    "drop_sparse": {
        "numeric": BASE_NUMERIC,
        "categorical": BASE_CATEGORICAL,
    },

    # missing 여부만 사용
    "presence_only": {
        "numeric": BASE_NUMERIC + ["deck_known"],
        "categorical": BASE_CATEGORICAL,
    },

    # deck category만 사용
    "deck_only": {
        "numeric": BASE_NUMERIC,
        "categorical": BASE_CATEGORICAL + ["deck_missing_category"],
    },

    # deck category + missing indicator
    "deck_presence": {
        "numeric": BASE_NUMERIC + ["deck_known"],
        "categorical": BASE_CATEGORICAL + ["deck_missing_category"],
    },
}


def split_data(df, seed):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["survived"],
        random_state=seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["survived"],
        random_state=seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_logistic_pipeline(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipe = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__",
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    model = LogisticRegression(
        max_iter=3000,
        random_state=SEED,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def run_logistic(
    train_df,
    val_df,
    test_df,
    variant,
    numeric_cols,
    categorical_cols,
):
    feature_cols = numeric_cols + categorical_cols

    model = build_logistic_pipeline(
        numeric_cols,
        categorical_cols,
    )

    model.fit(
        train_df[feature_cols],
        train_df["survived"],
    )

    val_pred = model.predict_proba(
        val_df[feature_cols]
    )[:, 1]

    test_pred = model.predict_proba(
        test_df[feature_cols]
    )[:, 1]

    val_auc = roc_auc_score(
        val_df["survived"],
        val_pred,
    )

    test_auc = roc_auc_score(
        test_df["survived"],
        test_pred,
    )

    test_ll = log_loss(
        test_df["survived"],
        test_pred,
    )

    return {
        "model": "Logistic",
        "variant": variant,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "test_logloss": test_ll,
    }


def prepare_lgbm_data(
    train_df,
    val_df,
    test_df,
    numeric_cols,
    categorical_cols,
):
    train = train_df.copy()
    val = val_df.copy()
    test = test_df.copy()

    # numeric:
    # LightGBM은 NaN 그대로 가능
    for c in numeric_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce")
        val[c] = pd.to_numeric(val[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")

    # categorical:
    # train category 기준으로 동일 code mapping
    for c in categorical_cols:
        tr = (
            train[c]
            .astype(object)
            .where(train[c].notna(), "__MISSING__")
            .astype(str)
        )

        va = (
            val[c]
            .astype(object)
            .where(val[c].notna(), "__MISSING__")
            .astype(str)
        )

        te = (
            test[c]
            .astype(object)
            .where(test[c].notna(), "__MISSING__")
            .astype(str)
        )

        categories = sorted(tr.unique().tolist())
        mapping = {
            v: i
            for i, v in enumerate(categories)
        }

        train[c] = tr.map(mapping).fillna(-1).astype(int)
        val[c] = va.map(mapping).fillna(-1).astype(int)
        test[c] = te.map(mapping).fillna(-1).astype(int)

    return train, val, test


def run_lightgbm(
    train_df,
    val_df,
    test_df,
    variant,
    numeric_cols,
    categorical_cols,
    seed,
):
    feature_cols = numeric_cols + categorical_cols

    train, val, test = prepare_lgbm_data(
        train_df,
        val_df,
        test_df,
        numeric_cols,
        categorical_cols,
    )

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        verbosity=-1,
    )

    model.fit(
        train[feature_cols],
        train["survived"],
        eval_set=[
            (
                val[feature_cols],
                val["survived"],
            )
        ],
        callbacks=[
            early_stopping(
                50,
                verbose=False,
            )
        ],
        categorical_feature=categorical_cols,
    )

    val_pred = model.predict_proba(
        val[feature_cols]
    )[:, 1]

    test_pred = model.predict_proba(
        test[feature_cols]
    )[:, 1]

    val_auc = roc_auc_score(
        val["survived"],
        val_pred,
    )

    test_auc = roc_auc_score(
        test["survived"],
        test_pred,
    )

    test_ll = log_loss(
        test["survived"],
        test_pred,
    )

    return {
        "model": "LightGBM",
        "variant": variant,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "test_logloss": test_ll,
        "best_iteration": model.best_iteration_,
    }


def run_single_seed(seed):
    df = load_data()
    df = add_sparse_features(df)

    train_df, val_df, test_df = split_data(
        df,
        seed,
    )

    print("\n" + "=" * 100)
    print(f"SEED = {seed}")
    print("=" * 100)

    deck_missing = train_df["deck"].isna().mean()

    print(
        f"Train deck missing rate: "
        f"{deck_missing:.4f} "
        f"({deck_missing * 100:.2f}%)"
    )

    results = []

    for variant, cfg in VARIANTS.items():
        print("\n" + "-" * 100)
        print(f"VARIANT: {variant}")
        print("-" * 100)

        numeric_cols = cfg["numeric"]
        categorical_cols = cfg["categorical"]

        log_result = run_logistic(
            train_df,
            val_df,
            test_df,
            variant,
            numeric_cols,
            categorical_cols,
        )

        print(
            f"Logistic | "
            f"AUC={log_result['test_auc']:.4f} "
            f"LogLoss={log_result['test_logloss']:.4f}"
        )

        results.append(log_result)

        lgbm_result = run_lightgbm(
            train_df,
            val_df,
            test_df,
            variant,
            numeric_cols,
            categorical_cols,
            seed,
        )

        print(
            f"LightGBM | "
            f"AUC={lgbm_result['test_auc']:.4f} "
            f"LogLoss={lgbm_result['test_logloss']:.4f}"
        )

        results.append(lgbm_result)

    return pd.DataFrame(results)


def summarize_multi_seed(all_results):
    summary = (
        all_results
        .groupby(
            ["model", "variant"],
            as_index=False,
        )
        .agg(
            mean_auc=("test_auc", "mean"),
            std_auc=("test_auc", "std"),
            mean_logloss=("test_logloss", "mean"),
            std_logloss=("test_logloss", "std"),
        )
        .sort_values(
            "mean_auc",
            ascending=False,
        )
    )

    return summary


def main(args):
    df = load_data()
    df = add_sparse_features(df)

    print_missing_summary(df)

    print(
        "\nSparse feature target:"
        "\n  deck = Cabin-derived information"
    )

    print(
        f"\nFull-data deck missing rate: "
        f"{df['deck'].isna().mean():.4f}"
    )

    all_results = []

    for seed in args.seeds:
        seed_df = run_single_seed(seed)
        seed_df["seed"] = seed
        all_results.append(seed_df)

    all_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    print("\n" + "=" * 120)
    print("ALL RESULTS")
    print("=" * 120)

    print(
        all_results
        .sort_values(
            ["seed", "test_auc"],
            ascending=[True, False],
        )
        .to_string(index=False)
    )

    summary = summarize_multi_seed(
        all_results
    )

    print("\n" + "=" * 120)
    print("MEAN ± STD")
    print("=" * 120)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    all_results.to_csv(
        "titanic_sparse_feature_all_results.csv",
        index=False,
    )

    summary.to_csv(
        "titanic_sparse_feature_summary.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        "  titanic_sparse_feature_all_results.csv"
    )
    print(
        "  titanic_sparse_feature_summary.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
    )

    args = parser.parse_args()

    main(args)
