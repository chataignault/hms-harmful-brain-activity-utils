import os
import pandas as pd
from typing import Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegressionCV

from .preamble import Grade, Dir, VOTE_COLS
from .process import (
    process_data_from_meta,
    process_extracted_features_to_design,
    extract_features_eeg,
    clean_covariates,
)


def train_GBC(
    train: pd.DataFrame,
    y_cols: str,
    max_nsample: Optional[int] = None,
    grade: Optional[Grade] = None,
    params: Optional[dict] = None,
    scale: bool = False,
) -> GradientBoostingClassifier:
    X, Y = process_data_from_meta(train, y_cols, max_nsample=max_nsample, grade=grade)
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    if params:  # grid search
        model = GradientBoostingClassifier(**params)
    else:  # set best params
        model = GradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.1,
            criterion="friedman_mse",
            n_estimators=200,
            max_depth=3,
        )
    X, Y = clean_covariates(X, Y)
    model.fit(X, Y)
    return model


def train_logistic_regression_CV(
    train: pd.DataFrame,
    y_cols: str,
    max_it: int = 100,
    grade: Optional[Grade] = None,
    max_nsample: Optional[int] = None,
    scale: bool = False,
    Cs: int = 10,
) -> LogisticRegressionCV:
    X, Y = process_data_from_meta(train, y_cols, max_nsample=max_nsample, grade=grade)
    model = LogisticRegressionCV(
        fit_intercept=True,
        penalty="l1",
        solver="saga",
        multi_class="multinomial",
        max_iter=max_it,
        Cs=Cs,
    )
    X, Y = clean_covariates(X, Y)
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    model.fit(X, Y)
    if scale:
        return model, scaler
    return model


def predict_probas_test_set(model, meta_test: pd.DataFrame) -> pd.DataFrame:
    predicted_probas_ = []
    for i in range(len(meta_test)):
        eeg_id = meta_test.loc[i, "eeg_id"]
        eeg_test = pd.read_parquet(os.path.join(Dir.eeg_test, f"{eeg_id}.parquet"))
        predicted_probas_sample = model.predict_proba(
            process_extracted_features_to_design([extract_features_eeg(eeg_test)])
        )
        predicted_probas_.append(
            pd.DataFrame(predicted_probas_sample, columns=VOTE_COLS, index=[eeg_id])
        )

    sub = pd.concat(predicted_probas_)
    sub.index.name = "eeg_id"
    return sub


def test_model(
    model: LogisticRegressionCV,
    y_cols: str,
    test_meta: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
):
    X, _ = process_data_from_meta(test_meta, y_cols, test_mode=True)
    X = X.fillna(0.0)  # OOPS, shuold not be NA values when computing the features
    if scaler:
        X = scaler.transform(X)
    return model.predict_proba(X)
