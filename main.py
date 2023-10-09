from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pandas as pd
import base64
import io
import numpy as np
import wave
import random
import argparse

# Create a FastAPI app
app = FastAPI()

def create_wav_in_memory(audio_data):
    """
    Create a WAV file in memory from audio data.

    Args:
        audio_data (pd.Series): Audio data as a pandas Series.

    Returns:
        str: Base64-encoded WAV audio.
    """
    global SAMPLING_RATE
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create a WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLING_RATE)  # Sample rate (adjust as needed)
        wf.writeframes(audio_data.tobytes())

    # Encode the WAV data as base64
    audio_base64 = base64.b64encode(wav_buffer.getvalue()).decode('utf-8')
    
    return audio_base64

def get_validation_form(request):
    """
    Generate the validation form data for rendering in the template.

    Args:
        request (Request): The FastAPI request object.

    Returns:
        TemplateResponse: The HTML template response.
    """
    # Randomly select a sample that hasn't been corrected yet
    index = next(indexes)
    
    num_completed = df[df['corrected'] != "<blank>"].shape[0]

    # Get the audio data from the DataFrame
    audio_data = df['audio'].loc[index]
    audio_base64 = create_wav_in_memory(audio_data)

    # Data to pass to the template
    template_data = {
        "num_completed": num_completed,
        "num_samples": num_samples,
        "audio_base64": audio_base64,
        "utterance": df['Utterance'].loc[index],
        "translation": df['translation'].loc[index],
        "corrected": df['corrected'].loc[index],
        "index": index,
    }

    return templates.TemplateResponse("validation_template.html", {"request": request, **template_data})

def parse_arguments():
    """
    Parse command-line arguments and return an argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(description="Transcript Correction Web App")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the FastAPI app on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI app on")
    parser.add_argument("--file-path", type=str, default="test.parquet", help="Path to the Parquet file")
    parser.add_argument("--templates", type=str, default="templates", help="Directory containing Jinja templates")
    parser.add_argument("--sampling-rate", type=int, default=16_000, help="Sampling rate of the audio")
    return parser.parse_args()



# 'validate' endpoint
@app.get("/")
async def validate(request: Request):
    """
    Endpoint to render the validation form.

    Args:
        request (Request): The FastAPI request object.

    Returns:
        TemplateResponse: The HTML template response.
    """
    response = get_validation_form(request)
    return response

# 'update' endpoint
@app.post("/")
async def update(request: Request, user_text: str = Form(...), index: int = Form(...)):
    """
    Endpoint to handle user text submissions and update the DataFrame.

    Args:
        request (Request): The FastAPI request object.
        user_text (str): User-submitted text.
        index (int): Index of the data row to update.

    Returns:
        TemplateResponse: The HTML template response.
    """
    # Handle the user's submitted text and index here
    # For demonstration, we'll just print them
    print(f"Received text: {user_text}, index: {index}")
    df['corrected'].loc[index] = user_text
    df.to_parquet(args.file_path)
    response = get_validation_form(request)

    return response

if __name__ == "__main__":
	import uvicorn
    
    # Parse command-line arguments
	args = parse_arguments()
	SAMPLING_RATE = args.sampling_rate

	# Create an instance of Jinja2Templates
	templates = Jinja2Templates(directory=args.templates)

	# Load your data here
	df = pd.read_parquet(args.file_path)

	indexes = iter(df[df['corrected']=="<blank>"].index)
	num_samples = df.shape[0]
	
	uvicorn.run(app, host=args.host, port=args.port)

