from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import pickle
import pandas as pd

FEATURES = pickle.load(open("models/features.pk", "rb"))

model = pickle.load(open("models/model.pk", "rb"))
column_equivalence = pickle.load(open("models/column_equivalence.pk", "rb"))


app = FastAPI()


def convert_numerical(features):
    """
    Converts the list of features from string to numerical values using the method to_numeric() of the module pandas

    Parameters
    ----------
    features : List[str]
        List with the following parameters in the same order: ['gender', 'product_amount', 'normalised_pop', 'age', 'work_age', 'registry_age', 'id_age', 'flag_work_phone',
         'region_score', 'region_city_score', 'flag_direperm_direcon_ciu', 'flag_direperm_diretra_ciu', 'external_score_1', 'external_score_2', 'external_score_3', 'num_lifts_average',
         'max_floor_average', 'max_floor_mode', 'num_lifts_median', 'max_floor_median', 'type_building', 'emergency_exits', 'age_mobilephone_days']

    """
    output = []
    for i, feat in enumerate(features):
        if i in column_equivalence:
            output.append(column_equivalence[i][feat])
        else:
            try:
                output.append(pd.to_numeric(feat))
            except:
                output.append(0)
    return output

class Data(BaseModel):
    data : List[str] = ['0']*23

@app.get("/")
def home():
    return {'message':'Hello!'}

@app.get("/query")
def query(feats : Data):
    features = convert_numerical(feats.data)
    response = {
        'response': [int(x) for x in model.predict([features])]
    }
    return response
    