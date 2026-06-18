from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

class Message(BaseModel):
    message:str

@app.post("/predict")
def predict(data: Message):

    vector = vectorizer.transform([data.message])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    if prediction == 1:
        return {
            "prediction":"Spam",
            "confidence":float(probability[1])
        }

    return {
        "prediction":"Not Spam",
        "confidence":float(probability[0])
    }