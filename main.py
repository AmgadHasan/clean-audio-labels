from fastapi import FastAPI, Request, Form, HTTPException
#from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import base64
import io
import numpy as np
import wave
import random
import argparse

app = FastAPI()

# Define command-line arguments
parser = argparse.ArgumentParser(description="Transcript Correction Web App")
parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI app on")
parser.add_argument("--file-path", type=str, default="test.parquet", help="Path to the Parquet file")
args = parser.parse_args()

# Create an instance of Jinja2Templates
templates = Jinja2Templates(directory="templates")

# Load your data here
df = pd.read_parquet(args.file_path)
indexes = iter(df.index)
num_samples = df.shape[0]

def get_validation_form(request):
    # Randomly select a sample that hasn't been corrected yet
    index = next(indexes)
    
    num_completed = df[df['corrected'] != "<blank>"].shape[0]

    # Get the audio data from the DataFrame
    audio_data = df['audio'].iloc[index]
    audio_data = (audio_data * 32767).astype(np.int16)

    # Create a WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(16000)  # Sample rate (adjust as needed)
        wf.writeframes(audio_data.tobytes())

    # Encode the WAV data as base64
    audio_base64 = base64.b64encode(wav_buffer.getvalue()).decode('utf-8')

    # Data to pass to the template
    template_data = {
        "num_completed": num_completed,
        "num_samples": num_samples,
        "audio_base64": audio_base64,
        "utterance": df['Utterance'].iloc[index],
        "translation": df['translation'].iloc[index],
        "corrected": df['corrected'].iloc[index],
        "index": index,
    }

    return templates.TemplateResponse("validation_template.html", {"request": request, **template_data})



# 'validate' endpoint
@app.get("/")
async def validate(request: Request):
    response = get_validation_form(request)
    return response
    


# 'update' endpoint
@app.post("/")
async def update(request: Request, user_text: str = Form(...), index: int = Form(...)):
    # Handle the user's submitted text and index here
    # For demonstration, we'll just print them
    print(f"Received text: {user_text}, index: {index}")
    df['corrected'].iloc[index] = user_text
    print(df['corrected'].iloc[index])
    df.to_parquet(file_path)
    response = get_validation_form(request)

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

