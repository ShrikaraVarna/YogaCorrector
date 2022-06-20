import json
import pyttsx3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers =  ["*"]
)

@app.post('/json')
async def post_data(finaldata : str):
    engine = pyttsx3.init() 
    rate = engine.getProperty("rate")  
    engine.setProperty("rate", 150 ) 
    engine.say(finaldata) 
    engine.runAndWait()
    return {"data" : "Posted successfully"}