"""Run the Churn FastAPI from the correct working directory."""
import os, sys
os.chdir(os.path.join(os.path.dirname(__file__), "churn-mlops"))
sys.path.insert(0, os.getcwd())
import uvicorn
uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
