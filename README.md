# IPL Next-Ball Predictor

## Setup (Windows PowerShell)
Run commands from the folder containing `app.py`, `train.py`, and `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

The training script uses the CSV files in `data/` and writes a model under `model/`.
It splits train/test by match ID to reduce match leakage and prints a classification report.
Predictions are estimates from historical data, not guaranteed outcomes or a live IPL feed.
