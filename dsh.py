import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from explainerdashboard import ClassifierExplainer, ExplainerDashboard


data = pd.read_csv('data/sdata.csv')
data = data.drop(data.columns[0], axis=1)

X = data.drop(data.columns[0], axis=1)
y = data.pop(data.columns[0])

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


model = pickle.load(open("models/model.pk", "rb"))

explainer = ClassifierExplainer(model, X_test, y_test)
ExplainerDashboard(explainer).run()