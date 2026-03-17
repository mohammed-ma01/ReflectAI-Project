import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv('dataset.csv')


numeric_cols = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') 
    df[col] = df[col].fillna(df[col].median())

text_cols = ['previous_day_mood', 'face_emotion_hint', 'reflection_quality']
for col in text_cols:
    df[col] = df[col].fillna('unknown')


df['time_numeric'] = df['time_of_day'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
df['time_numeric'] = df['time_numeric'].fillna(0) 

df['time_period'] = df['time_of_day'].astype(str).str.extract(r'([a-zA-Z]+)', expand=False).str.lower()


typo_fixes = {
    'moming': 'morning',
    'maming': 'morning',
    'marring': 'morning',
    'attemoon': 'afternoon',
    'nicht': 'night'
}
df['time_period'] = df['time_period'].replace(typo_fixes)
df['time_period'] = df['time_period'].fillna('unknown')

df = df.drop(columns=['time_of_day'])


df.to_csv('dataset_cleaned.csv', index=False)

print("\nCleaning complete! Saved as 'dataset_cleaned.csv'.")
print("\n--- Any missing values left? ---")
print(df.isnull().sum())