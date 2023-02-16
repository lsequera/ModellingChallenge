import pickle
import pandas as pd

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
from explainerdashboard import ClassifierExplainer, ExplainerDashboard

# Loading pre-processed data
data = pd.read_csv('data/sdata.csv')

# Dropping the index column
data = data.drop(data.columns[0], axis=1)

# Filtering target
X = data.drop(data.columns[0], axis=1)
y = data.pop(data.columns[0])

# Balacing the data
over_sampler = SMOTE(random_state=42)
X_res, y_res = over_sampler.fit_resample(X, y)

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Loading the pre-trained XGBoost model
model = pickle.load(open("models/model.pk", "rb"))

# Creating and running the interactive dashboard
explainer = ClassifierExplainer(model, X_test, y_test)
ExplainerDashboard(explainer).run()
