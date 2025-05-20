import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import make_scorer, roc_auc_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real,Integer

wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target
print(wine_df.dtypes)
print(wine_df.isna().sum())

y = wine_df['target']
X = wine_df.drop(columns = ['target'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.2, stratify=y, random_state = 42)

sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)
cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
roc_auc_ovr = make_scorer(roc_auc_score, multi_class = 'ovr', needs_proba = True)
param_space = {
    "n_estimators" : Integer(100,200),
    "learning_rate": Real(0.005, 0.2),
    "max_depth" : Integer(3,10),
    "min_child_weight" : Integer(1,4),
    "reg_alpha" : Real(0,1),
    "reg_lambda" : Real(0,5)
}

xgb = XGBClassifier(n_estimators = 100, objective = 'multi:softprob', random_state = 42, eval_metric = 'merror', tree_method = 'hist')

bayes_search = BayesSearchCV(
    estimator = xgb,
    search_spaces = param_space,
    n_iter = 115,
    scoring = 'roc_auc_ovr',
    cv = cv,
    random_state = 42,
    refit = True
)

bayes_search.fit(X_train, y_train, sample_weight = sample_weights)

for param, val in bayes_search.best_params_.items():
    print(F"{param}: {val}")

best_model = bayes_search.best_estimator_

y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = best_model.classes_)
disp.plot()


