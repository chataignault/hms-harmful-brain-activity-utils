import numpy as np
import pandas as pd
from typing import Optional, Callable, Union
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingRegressor,
)

from .preamble import Grade, VOTE_COLS
from .process import process_data_from_meta, print_summary_metadata
from .classes import FeatureGenerator
from .scoring import score


def train_GBC(
    train: pd.DataFrame,
    feature_generator: Callable,
    y_cols: str,
    max_nsample: Optional[int] = None,
    grade: Optional[Grade] = None,
    params: Optional[dict] = None,
    scale: bool = False,
) -> GradientBoostingClassifier:
    X, Y = process_data_from_meta(
        train, feature_generator, y_cols, max_nsample=max_nsample, grade=grade
    )
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
    model.fit(X, np.ravel(Y))
    if scale:
        return model, scaler
    return model


def train_GBRegressors(
    train: pd.DataFrame,
    feature_generator: Callable,
    y_cols: str,
    max_nsample: Optional[int] = None,
    grade: Optional[Grade] = None,
    params: Optional[dict] = None,
    scale: bool = False,
) -> GradientBoostingRegressor:
    """
    Attempt to train one tree for each category :
    The idea is that classification looses information on what other classes were considered in the votes
    """
    X, Y = process_data_from_meta(
        train,
        feature_generator,
        y_cols,
        max_nsample=max_nsample,
        grade=grade,
        classification=False,
    )
    Y = Y.values
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    if params:  # grid search
        models = [
            GradientBoostingRegressor(loss="absolute_error", criterion="friedman_mse", **params)
            for _ in range(len(Y[0]))
        ]
    else:  # set best params
        models = [
            GradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.1,
                criterion="friedman_mse",
                n_estimators=200,
                max_depth=3,
            )
            for _ in range(len(Y[0]))
        ]
    models = [model.fit(X, Y[:, i]) for i, model in enumerate(models)]
    if scale:
        return models, scaler
    return models


def train_random_forest_classifier(
    train: pd.DataFrame,
    feature_generator: Callable,
    y_cols: str,
    max_nsample: Optional[int] = None,
    grade: Optional[Grade] = None,
    params: Optional[dict] = None,
    scale: bool = False,
) -> RandomForestClassifier:
    X, Y, selected_meta = process_data_from_meta(
        train, feature_generator, y_cols, max_nsample=max_nsample, grade=grade
    )
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    if params:  # grid search
        model = RandomForestClassifier(**params)
    else:  # set best params
        model = RandomForestClassifier(
            criterion="gini",
            max_depth=None,
            max_features="sqrt",
            min_samples_split=2,
            bootstrap=True,
            oob_score=True,
            n_estimators=300,
        )
    model.fit(X, Y)

    display_in_sample_score(X, selected_meta, model)
    if scale:
        return model, scaler
    return model


def train_logistic_regression_CV(
    train: pd.DataFrame,
    feature_generator: Callable,
    y_cols: str,
    max_it: int = 10000,
    grade: Optional[Grade] = None,
    max_nsample: Optional[int] = None,
    scale: bool = False,
    Cs: int = 10,
    fit_intercept: bool = False,
) -> LogisticRegressionCV:
    X, Y, selected_meta = process_data_from_meta(
        train, feature_generator, y_cols, max_nsample=max_nsample, grade=grade
    )

    model = LogisticRegressionCV(
        fit_intercept=fit_intercept,
        penalty="l1",
        solver="saga",  # good for large datasets
        multi_class="multinomial",
        max_iter=max_it,
        Cs=Cs,
    )
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    model.fit(X, np.ravel(Y))

    display_in_sample_score(X, selected_meta, model)

    if scale:
        return model, scaler

    return model


def display_in_sample_score(
    X: np.ndarray,
    selected_meta: pd.DataFrame,
    model: Union[LogisticRegressionCV, GradientBoostingClassifier],
):
    """
    Compare the in sample score with :
        - uniform prediction
        - argmax prediction
    """
    print_summary_metadata(selected_meta)
    pp = [1.0 / 6] * 6
    uniform_in_score = score(
        np.array([pp for _ in range(len(selected_meta))]), selected_meta[VOTE_COLS].values
    )
    print(">>> Uniform perdiction")
    display(uniform_in_score)
    in_predicted_probas = model.predict_proba(X)
    in_sample_score = score(in_predicted_probas, selected_meta[VOTE_COLS].values)
    print(">>> Model predicted probabilities score")
    display(in_sample_score)
    max_proba_predict = (
        in_predicted_probas == np.repeat(np.max(in_predicted_probas, axis=1), 6).reshape((-1, 6))
    ).astype(float)
    print(">>> Argmax prediction")
    display(score(max_proba_predict, selected_meta[VOTE_COLS].values))


def test_model(
    model: Union[LogisticRegressionCV, GradientBoostingClassifier],
    feature_generator: FeatureGenerator,
    y_cols: str,
    test_meta: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    classification: bool = True,
):
    X, *other = process_data_from_meta(
        test_meta, feature_generator, y_cols, classification=classification
    )
    if scaler:
        X = scaler.transform(X)
    if classification:
        return model.predict_proba(X)
    else:
        Y_hat = model.predict(X)  # logodds
        eY_hat = np.exp(Y_hat)
        return eY_hat


def predict_probas_test_set(
    model, meta_test: pd.DataFrame, feature_generator: FeatureGenerator
) -> pd.DataFrame:
    """
    Iterate on the test files, stack predicted probabilities with eeg_id as index
    Take model and adapted feature generator object
    """
    predicted_probas_ = []
    for i in range(len(meta_test)):
        eeg = meta_test.iloc[i]
        features = feature_generator.process(meta_test)
        predicted_probas_sample = model.predict_proba(features)

        predicted_probas_.append(
            pd.DataFrame(predicted_probas_sample, columns=VOTE_COLS, index=[eeg["eeg_id"]])
        )
    sub = pd.concat(predicted_probas_)
    sub.index.name = "eeg_id"
    return sub
