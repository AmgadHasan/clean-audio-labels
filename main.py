from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import base64
import io
import numpy as np
import wave
import random

app = FastAPI()

file_path="test.parquet"

df = pd.read_parquet(file_path)
indexes = iter(df.index)
num_samples = df.shape[0]

def get_validation_form():
    # Randomly a sample that hasn't been corrected yet
    #index = random.choice(df[df['corrected'] != "<blank>"].index)
    index = next(indexes)
    #index = random.choice(df.index)
    
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


    # HTML form for the 'validate' endpoint
    validate_form = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Validate Endpoint</title>
    </head>
    <body>
        <h1>Validate Endpoint</h1>
        <h2>Completed so far: {num_completed}/{num_samples}</h2>
        <audio controls>
            <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        <p>Utterance: {df['Utterance'].iloc[index]}</p>
        <p>Translation: {df['translation'].iloc[index]}</p>
        <p>Corrected: {df['corrected'].iloc[index]}</p>
        <p>Index: {index}</p>
        <form action="/" method="post">
            <label for="user_text">Enter Text:</label>
            <!-- Pre-fill the text box with translation_value -->
            <input type="text" id="user_text" name="user_text" required style="width: 70%;" value="{df['corrected'].iloc[index]}">
            <input type="hidden" name="index" value="{index}">
            <input type="submit" value="Submit">
        </form>
    </body>
    </html>
    """

    return HTMLResponse(content=validate_form)



# 'validate' endpoint
@app.get("/", response_class=HTMLResponse)
async def validate():
    response = get_validation_form()
    return response
    


# 'update' endpoint
@app.post("/")
async def update(user_text: str = Form(...), index: int = Form(...)):
    # Handle the user's submitted text and index here
    # For demonstration, we'll just print them
    print(f"Received text: {user_text}, index: {index}")
    df['corrected'].iloc[index] = user_text
    print(df['corrected'].iloc[index])
    df.to_parquet(file_path)
    response = get_validation_form()

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

