# 🛡️ BotGuard — Bot-Generated Comment Detection

Deep learning model to detect bot-generated comments on social media.

## Model
BiLSTM + Attention (PyTorch) — two-branch architecture combining
comment text (BiLSTM) and engineered linguistic features (Dense).

## Dataset
YouTube Spam Collection — UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/380/youtube+spam+collection

## Reference
Airlangga, G. (2024). Spam Detection in YouTube Comments Using Deep
Learning Models. *MALCOM Journal*, 4(4), 1533–1538.
https://doi.org/10.57152/malcom.v4i4.1671

## Setup

```bash
pip install -r requirements.txt
python -m src.train
streamlit run app/streamlit_app.py
```