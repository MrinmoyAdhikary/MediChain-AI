# MediChain-AI

Create a virtual environment:

In powershell:

 `python -m venv .venv`

 `.venv\Scripts\Activate.ps1`


In cmd:
 `.venv\Scripts\activate`

Install Dependecies:
``` 
    python -m pip install --upgrade pip
    pip install -r requirements.txt  
```

Train the model (command in terminal):
 `python scripts/train.py`

REQUIREMENTS:
   - tensorflow==2.21.0
   - keras==3.15.1
   - numpy
   - pillow