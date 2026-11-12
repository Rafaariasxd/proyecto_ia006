import joblib
import numpy as np
import pandas as pd

MONTHS = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}

def load_model(path):
  return joblib.load(path)

def prepare(df, art):
  df = df.copy()

  df['meal'] = df['meal'].replace('Undefined', 'SC')
  df['children'] = pd.to_numeric(df['children'], errors='coerce').fillna(0).astype(int)
  df['agent'] = pd.to_numeric(df['agent'], errors='coerce').fillna(0).astype(int).astype(str).replace('0', 'None')
  df['country'] = df['country'].fillna('Unknown')
  df['adr'] = df['adr'].clip(art['adr_bounds'][0], art['adr_bounds'][1])

  df['arrival_month'] = df['arrival_date_month'].map(MONTHS)
  date = pd.to_datetime(pd.DataFrame({'year': df['arrival_date_year'], 'month': df['arrival_month'], 'day': df['arrival_date_day_of_month']}))
  df['arrival_dow'] = date.dt.dayofweek
  df['arrival_weekend'] = (df['arrival_dow'] >= 5).astype(int)
  df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
  df['total_guests'] = df['adults'] + df['children'] + df['babies']
  df['has_children'] = ((df['children'] + df['babies']) > 0).astype(int)

  df['country'] = df['country'].where(df['country'].isin(art['top_country']), 'Other')
  df['agent'] = df['agent'].where(df['agent'].isin(art['top_agent']), 'Other')

  for col in art['log_cols']:
    df[col] = np.log1p(df[col])

  df[['arrival_month', 'arrival_dow']] = df[['arrival_month', 'arrival_dow']].astype(str)
  X = pd.get_dummies(df[art['numerical'] + art['binary'] + art['categorical']], columns=art['categorical'], dtype=int)
  X = X.reindex(columns=art['columns'], fill_value=0)
  X[art['numerical']] = art['scaler'].transform(X[art['numerical']])

  if art['reduction'] == 'PCA':
    X = art['pca'].transform(X)
  elif art['reduction'] == 'LDA':
    X = art['lda'].transform(X)
  return X

def predict(df, art):
  X = prepare(df, art)
  return art['model'].predict_proba(X)[:, 1]
