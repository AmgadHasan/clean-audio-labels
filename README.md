# Clean Audio Data Web App

This is a web application built with FastAPI that allows users to correct transcripts and validate them against audio recordings. The application uses Jinja2 templates for rendering the user interface and provides an easy-to-use interface for correcting and validating transcripts.

## Features

- Render a user-friendly web interface for correcting and validating transcripts.
- Play audio recordings associated with each transcript.
- Dynamically load data from a Parquet file.
- Easily configurable via command-line arguments.
- Modular code organization for maintainability.

## Installation
### 1. Create a virtual environment
```
python -m venv .venv
```
### 2. Install dependencies

Before running this application, make sure you have the following prerequisites installed:

- Python 3.6 or higher
- FastAPI
- Jinja2
- pandas
- numpy
- wave

You can install these dependencies using `pip`:

```
pip install -r requirements.txt
```

## Running the webapp
To run the webapp, make sure you have a parquet file that contains the transcripts and audio

```
python main.py --port 8000 --host 0.0.0.0
```
