import numpy as np
import pandas as pd


LEAKAGE_COLS = ['reservation_status', 'reservation_status_date']


def load_raw(path='data/hotels.csv'):
    return pd.read_csv(path)


def clean(df):
    df = df.drop(columns=LEAKAGE_COLS)
    df = df.drop_duplicates().reset_index(drop=True)

    total_people = df['adults'] + df['children'].fillna(0) + df['babies']
    df = df[total_people > 0].reset_index(drop=True)

    df = df[(df['adr'] >= 0) & (df['adr'] < 1000)].reset_index(drop=True)

    df['children'] = df['children'].fillna(0).astype(int)
    df['country'] = df['country'].fillna('Unknown')
    df['agent'] = df['agent'].fillna(0).astype(int)
    df['has_company'] = df['company'].notna().astype(int)
    df = df.drop(columns=['company'])

    return df
