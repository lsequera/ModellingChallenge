from fastapi import FastAPI
from pydantic import BaseModel

import pickle

model = pickle.load(open("churn/models/model.pk", "rb"))


app = FastAPI()

# class Data(BaseModel):


@app.get("/")
def home():
    return {'message':'Hello!'}
    