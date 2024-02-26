"""








#Please download the saved model from the given link

#https://drive.google.com/uc?export=download&id=1xx3b0HBp8W5OMmHxSLfkjrY5dghcJs67









"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
import uvicorn
import re
import string
import safetensors
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


app = FastAPI()


@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


model_dir = "pegasus-samsum-model"
tokeniser_dir = "tokenizer"


model = PegasusForConditionalGeneration.from_pretrained(model_dir)
tokeniser = PegasusTokenizer.from_pretrained(tokeniser_dir)


class InputData(BaseModel):
    text: str = "            Please enter the text you want to summarise             "


@app.post('/predict')
async def summarize_text(data: InputData):
    inputs = tokeniser([data.text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs.input_ids)
    summary = tokeniser.decode(summary_ids[0], skip_special_tokens=True)
    return {'summary': summary}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
