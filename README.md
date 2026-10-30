# Meeting Notes AI

## Quick setup

Before starting, install and open **Docker Desktop**. You also need Python 3.10+
and Git.

Run these commands in **PowerShell from the repository folder**:

### 1. Install the registration dependencies

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\scripts\requirements.txt
```

### 2. Start MLflow

```powershell
docker compose up -d mlflow
```

Wait until MLflow is ready, then open <http://localhost:5050>.

### 3. Register the models

Run both commands. The first run downloads large model files and may take a
while.

```powershell
python .\scripts\register_pretrained.py whisper-asr whisper-base --alias champion
python .\scripts\register_pretrained.py summarizer-distilbart distilbart-cnn-12-6 --alias champion
```

### 4. Start the API

```powershell
docker compose up -d --build api
```

The API may take a few minutes to load the models. Check that it is ready:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

When the response shows `"status": "ok"`, open the app at
<http://localhost:8000>.

## If something goes wrong

View the logs:

```powershell
docker compose logs -f
```

Stop the containers:

```powershell
docker compose down
```

If PowerShell blocks the virtual environment activation, run this once in the
same terminal and retry activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
