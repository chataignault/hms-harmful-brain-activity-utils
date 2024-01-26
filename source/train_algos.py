import os
import pandas as pd
from typing import Optional
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from .preamble import Grade, Dir, VOTE_COLS
from .process import process_data_from_meta, process_extracted_features_to_design, extract_features_eeg


def train_GBC(train:pd.DataFrame, y_cols:str, max_it:Optional[int]=None, grade:Optional[Grade]=None) -> GradientBoostingClassifier:
    X, Y = process_data_from_meta(train, y_cols, max_it=max_it, grade=grade)
    model = GradientBoostingClassifier()
    model.fit(X, Y)
    return model

def train_logistic_regression_CV(train:pd.DataFrame, y_cols:str, max_it:Optional[int]=None, grade:Optional[Grade]=None) -> LogisticRegressionCV:
    X, Y = process_data_from_meta(train, y_cols, max_it=max_it, grade=grade)
    model = LogisticRegressionCV(fit_intercept=True, penalty="l1", solver="saga")
    print(X.shape, Y.shape)
    model.fit(X, Y)
    return model

def predict_probas_test_set(model, meta_test:pd.DataFrame) -> pd.DataFrame:
    predicted_probas_ = []
    for i in range(len(meta_test)):
        eeg_id = meta_test.loc[i, "eeg_id"]
        eeg_test = pd.read_parquet(os.path.join(Dir.eeg_test, f"{eeg_id}.parquet"))
        predicted_probas_sample = model.predict_proba(process_extracted_features_to_design([extract_features_eeg(eeg_test)]))
        predicted_probas_.append(pd.DataFrame(predicted_probas_sample, columns=VOTE_COLS, index=[eeg_id]))

    sub = pd.concat(predicted_probas_)
    sub.index.name = "eeg_id"
    return sub

def test_model(model:LogisticRegressionCV, y_cols:str, test_meta:pd.DataFrame):
    X, Y = process_data_from_meta(test_meta, y_cols)
    return model.predict_proba(X)
