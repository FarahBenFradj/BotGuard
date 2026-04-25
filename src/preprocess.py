import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re, json, glob, os

TEXT_COL  = 'CONTENT'
LABEL_COL = 'CLASS'
MAX_LEN   = 128
MAX_VOCAB = 15_000

ENGINEERED = [
    'text_len', 'word_count', 'exclamation_count',
    'question_count', 'caps_ratio', 'digit_ratio',
    'url_count', 'has_url', 'repeated_chars', 'unique_word_ratio'
]


def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'http\S+', '<url>', text)
    text = re.sub(r'@\w+',   '<user>', text)
    text = re.sub(r'#\w+',   '<hashtag>', text)
    text = re.sub(r'[^a-z0-9\s<>!?]', ' ', text)
    return text.strip()


def engineer_features(df):
    raw = df[TEXT_COL].fillna('').astype(str)
    df['text_len']          = raw.apply(len)
    df['word_count']        = raw.apply(lambda x: len(x.split()))
    df['exclamation_count'] = raw.apply(lambda x: x.count('!'))
    df['question_count']    = raw.apply(lambda x: x.count('?'))
    df['caps_ratio']        = raw.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1))
    df['digit_ratio']       = raw.apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1))
    df['url_count']         = raw.apply(lambda x: x.lower().count('http'))
    df['has_url']           = (df['url_count'] > 0).astype(int)
    df['repeated_chars']    = raw.apply(
        lambda x: sum(1 for i in range(1, len(x)) if x[i] == x[i-1]) / max(len(x), 1))
    df['unique_word_ratio'] = raw.apply(
        lambda x: len(set(x.split())) / max(len(x.split()), 1))
    return df


class Tokenizer:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}

    def fit(self, texts):
        from collections import Counter
        counter = Counter(w for t in texts for w in t.split())
        for word, _ in counter.most_common(MAX_VOCAB - 2):
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

    def encode(self, text):
        ids = [self.word2idx.get(w, 1) for w in text.split()[:MAX_LEN]]
        ids += [0] * (MAX_LEN - len(ids))
        return ids

    def save(self, path):
        with open(path, 'w') as f: json.dump(self.word2idx, f)

    @classmethod
    def load(cls, path):
        t = cls()
        with open(path) as f: t.word2idx = json.load(f)
        return t

    def __len__(self): return len(self.word2idx)


class BotDataset(Dataset):
    def __init__(self, texts, meta, labels):
        self.texts  = torch.LongTensor(texts)
        self.meta   = torch.FloatTensor(meta)
        self.labels = torch.FloatTensor(labels)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return self.texts[i], self.meta[i], self.labels[i]


def load_data(data_dir='data/youtube', test_size=0.15, val_size=0.15):
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'.\n"
            "Download from: https://archive.ics.uci.edu/dataset/380/youtube+spam+collection"
        )

    dfs = []
    for f in csv_files:
        try:
            tmp = pd.read_csv(f, encoding='latin-1')
            tmp.columns = [c.upper().strip() for c in tmp.columns]
            if 'CONTENT' in tmp.columns and 'CLASS' in tmp.columns:
                dfs.append(tmp[['CONTENT', 'CLASS']])
                print(f"  ✅ {os.path.basename(f)} — {len(tmp)} rows")
        except Exception as e:
            print(f"  ⚠️  {f}: {e}")

    df = pd.concat(dfs, ignore_index=True).dropna(subset=['CONTENT', 'CLASS'])
    df['CLASS'] = pd.to_numeric(df['CLASS'], errors='coerce').fillna(0).astype(int)
    print(f"\nTotal: {len(df)} comments")
    print(df['CLASS'].value_counts().rename({0: 'Human', 1: 'Bot/Spam'}))

    df = engineer_features(df)
    df['clean_text'] = df[TEXT_COL].apply(clean_text)

    avail_meta = [c for c in ENGINEERED if c in df.columns]
    for col in avail_meta:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    tok = Tokenizer()
    tok.fit(df['clean_text'].tolist())
    X_text = np.array([tok.encode(t) for t in df['clean_text']])

    scaler = StandardScaler()
    X_meta = scaler.fit_transform(df[avail_meta].values)

    y = df['CLASS'].values

    X_tr, X_tmp, M_tr, M_tmp, y_tr, y_tmp = train_test_split(
        X_text, X_meta, y, test_size=test_size+val_size,
        random_state=42, stratify=y)
    X_val, X_te, M_val, M_te, y_val, y_te = train_test_split(
        X_tmp, M_tmp, y_tmp,
        test_size=test_size/(test_size+val_size),
        random_state=42, stratify=y_tmp)

    print(f"Train: {len(y_tr)} | Val: {len(y_val)} | Test: {len(y_te)}")
    return (
        BotDataset(X_tr, M_tr, y_tr),
        BotDataset(X_val, M_val, y_val),
        BotDataset(X_te, M_te, y_te),
        tok, scaler, avail_meta
    )