from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def feature_selection_scores(dataset=df_train, rf_classifier=None):
    
    X_train_ = dataset.drop(columns=['g0_arrest_1'])
    
    y_train = dataset['g0_arrest_1']

    scores = np.zeros(shape=X_train_.shape[1])

    # X_train_ = RobustScaler().fit_transform(X_train_)

    for i in tqdm(range(100)):    
        # Add random vector to training data
        random_vector       = np.random.rand(X_train_.shape[0], 1) # random samples from an uniform distribution over [0, 1)
        # random_vector = np.random.normal(loc=0, scale=0.1, size=(X_train_.shape[0], 1))

        X_train_with_random = np.concatenate((X_train_, random_vector), axis=1)

        if rf_classifier is None:
            rf_classifier = RandomForestClassifier(random_state=i) # Use default RF classifier

        rf_classifier.fit(X_train_with_random, y_train)

        feature_importances = rf_classifier.feature_importances_
        # pdb.set_trace()
        # Random vector feature importance
        threshold = feature_importances[-1]

        scores[np.argwhere(feature_importances > threshold)] += 1

    return scores

scores = feature_selection_scores(dataset=df_train, rf_classifier=xgb.XGBClassifier())
