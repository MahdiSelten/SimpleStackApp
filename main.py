
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np


from pydantic import BaseModel

app = FastAPI()


model = joblib.load('SVCModel')


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://simplestackappfront.onrender.com", "http://simplestackapp.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

items = [{"id": 1, "name": "Example item"}]





class PassengerInput(BaseModel):
    pclass: int
    sex: str
    sibsp: int
    parch: int
    fare: float


def preprocess_input(input: PassengerInput):
    sex_encoded = 1 if input.sex.lower == "female" else 0
    return [
        input.pclass,
        sex_encoded,
        input.sibsp,
        input.parch,
        input.fare

    ]



@app.get("/items")
def get_items():
    return items

@app.post("/items")
def create_item(item: dict):
    item["id"] = len(items) + 1
    items.append(item)
    return item


@app.post("/predict")
def handle_prediction(pred:PassengerInput):

    processedData = preprocess_input(pred)
    processedData = np.array(processedData).reshape(1, -1)
    prediction = model.predict(processedData)

    return {"prediction": int(prediction[0])}

