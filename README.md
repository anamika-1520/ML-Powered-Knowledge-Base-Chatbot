# Car Price Chatbot

A production-ready repo for an Indian car-price prediction and car-knowledge chatbot.

## Included files

- `ml_tool.py` — ML inference helper for used-car price prediction.
- `rag_faiss.py` — FAISS-based RAG knowledge engine for car-domain guidance.
- `graph.py` — LangGraph routing logic and chatbot orchestration.
- `app.py` — Streamlit UI for the chatbot.
- `car_price_predictor.ipynb` — Notebook with interactive experimentation and example usage.

## Requirements

Install the required Python packages:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Setup

1. Ensure the trained model file is available in the repo root:
   - `LinearRegressionModel.pkl`

2. If your LangGraph / Groq setup requires environment variables, create a `.env` file and add the required keys.

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Notes

- The ML predictor expects:
  - `name`
  - `company`
  - `year`
  - `kms_driven`
  - `fuel_type`

- The RAG path uses text embeddings and FAISS to answer broader car-domain questions.
- `graph.py` routes user messages automatically between ML prediction, RAG knowledge, and direct chat responses.

## GitHub repository setup

1. Initialize Git:

```bash
git init
git add .
git commit -m "Initial commit"
```

2. Create a GitHub repo and push:

```bash
gh repo create <your-username>/<repo-name> --public --source=. --remote=origin
git push -u origin main
```

If you do not have `gh`, create the repository on GitHub and then run:

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Recommended improvements

- Add `LinearRegressionModel.pkl` or update `ml_tool.py` to download it from a storage location.
- Add a `.env.example` file if you need to document required API keys.
- Add `tests/` for unit tests around the routing and prediction logic.
