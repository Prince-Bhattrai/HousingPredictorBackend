from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel, Field
from typing import Annotated
import pickle
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()


origins = [
    "http://localhost:5173",
    "https://housing-predictor-ebon.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,
    allow_methods=["*"],         
    allow_headers=["*"],        
)


with open("regressor_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

class InputData(BaseModel):
    area: Annotated[int, Field(..., description="Enter area of house.")]
    bedrooms: Annotated[int, Field(..., description="Number of bedrooms.")]
    bathrooms: Annotated[int, Field(..., description="Number of bathrooms.")]
    parking: Annotated[int, Field(..., description="Number of parking spaces.")]
    mainroad: Annotated[str, Field(..., description="Is main road available?")]
    guestroom: Annotated[str, Field(..., description="Is guest room available?")]
    basement: Annotated[str, Field(..., description="Is basement available?")]
    hotwaterheating: Annotated[str, Field(..., description="Is hot water heating available?")]
    airconditioning: Annotated[str, Field(..., description="Is air conditioning available?")]

@app.get("/")
def home():
    return JSONResponse(status_code=200, content={
        "success": True,
        "message": "House price prediction API is running..."
    })

@app.post("/predict")
def predict(data: InputData):
    try:
        predict_data = pd.DataFrame([{
            "area": data.area,
            "bedrooms": data.bedrooms,
            "bathrooms": data.bathrooms,
            "parking": data.parking,
            "mainroad": data.mainroad,
            "guestroom": data.guestroom,
            "basement": data.basement,
            "hotwaterheating": data.hotwaterheating,
            "airconditioning": data.airconditioning
        }])
        predicted_price = loaded_model.predict(predict_data)
        return JSONResponse(status_code=200, content={
            "success": True,
            "message": "Predicted price of house",
            "price": predicted_price.tolist()
        })
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Server error, please try again later.")
