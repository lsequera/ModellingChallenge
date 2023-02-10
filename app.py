from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# class Data(BaseModel):


@app.get("/")
def home():
    return {'message':'Hello!'}
    