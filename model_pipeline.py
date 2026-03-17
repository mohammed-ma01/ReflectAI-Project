import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

print("Loading data...")
train_df = pd.read_csv('dataset_cleaned.csv')
test_df = pd.read_csv('test_dataset.csv') 

print("Cleaning test data...")
numeric_cols = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
for col in numeric_cols:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce') 
    test_df[col] = test_df[col].fillna(train_df[col].median())

text_cols = ['previous_day_mood', 'face_emotion_hint', 'reflection_quality']
for col in text_cols:
    test_df[col] = test_df[col].fillna('unknown')

test_df['time_numeric'] = test_df['time_of_day'].astype(str).str.extract(r'(\d+)', expand=False).astype(float)
test_df['time_numeric'] = test_df['time_numeric'].fillna(0) 

test_df['time_period'] = test_df['time_of_day'].astype(str).str.extract(r'([a-zA-Z]+)', expand=False).str.lower()
typo_fixes = {'moming': 'morning', 'maming': 'morning', 'marring': 'morning', 'attemoon': 'afternoon', 'nicht': 'night'}
test_df['time_period'] = test_df['time_period'].replace(typo_fixes).fillna('unknown')

y_state_train = train_df['emotional_state']
y_int_train = pd.to_numeric(train_df['intensity'], errors='coerce').fillna(3)

label_encoder = LabelEncoder()
y_state_train_encoded = label_encoder.fit_transform(y_state_train)

print("Vectorizing and Encoding...")

tfidf = TfidfVectorizer(max_features=500, stop_words='english')
text_train = tfidf.fit_transform(train_df['journal_text'].fillna('')).toarray()
text_test = tfidf.transform(test_df['journal_text'].fillna('')).toarray()

text_train_df = pd.DataFrame(text_train, columns=[f"word_{w}" for w in tfidf.get_feature_names_out()])
text_test_df = pd.DataFrame(text_test, columns=[f"word_{w}" for w in tfidf.get_feature_names_out()])

categorical_cols = ['ambience_type', 'previous_day_mood', 'face_emotion_hint', 'reflection_quality', 'time_period']
meta_train_df = pd.get_dummies(train_df[categorical_cols])
meta_test_df = pd.get_dummies(test_df[categorical_cols])

meta_train_df, meta_test_df = meta_train_df.align(meta_test_df, join='left', axis=1, fill_value=0)

numeric_cols_final = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level', 'time_numeric']
num_train_df = train_df[numeric_cols_final]
num_test_df = test_df[numeric_cols_final]

X_train = pd.concat([text_train_df, meta_train_df, num_train_df], axis=1)
X_test = pd.concat([text_test_df, meta_test_df, num_test_df], axis=1)

print("Training models...")
state_model = xgb.XGBClassifier(random_state=42)
state_model.fit(X_train, y_state_train_encoded)

intensity_model = xgb.XGBRegressor(random_state=42)
intensity_model.fit(X_train, y_int_train)

print("Generating final predictions...")
test_state_preds_encoded = state_model.predict(X_test)
test_state_probs = state_model.predict_proba(X_test)
test_int_preds = intensity_model.predict(X_test)

predicted_states = label_encoder.inverse_transform(test_state_preds_encoded)
confidence_scores = np.max(test_state_probs, axis=1)
uncertain_flags = (confidence_scores < 0.45).astype(int)


def decision_engine(row):
    state = row['predicted_state']
    intensity = row['predicted_intensity']
    time_period = row['time_period']
    stress = row['stress_level']
    
    what_to_do = "pause"
    when_to_do = "later_today"
    
    if state in ['overwhelmed', 'restless'] or stress >= 4:
        if intensity >= 4:
            what_to_do = "box_breathing"
            when_to_do = "now"
        else:
            what_to_do = "journaling"
            when_to_do = "within_15_min"
            
    elif state == 'focused':
        what_to_do = "deep_work"
        when_to_do = "now" if time_period in ['morning', 'afternoon'] else "tomorrow_morning"
        
    elif state == 'calm' and time_period in ['evening', 'night']:
        what_to_do = "rest"
        when_to_do = "tonight"
        
    elif state == 'mixed':
        what_to_do = "grounding"
        when_to_do = "within_15_min"
        
    return pd.Series([what_to_do, when_to_do])


final_df = pd.DataFrame({
    'id': test_df['id'],
    'predicted_state': predicted_states,
    'predicted_intensity': np.clip(np.round(test_int_preds), 1, 5).astype(int), 
    'confidence': np.round(confidence_scores, 2),
    'uncertain_flag': uncertain_flags,
    'time_period': test_df['time_period'],   
    'stress_level': test_df['stress_level']  
})

final_df[['what_to_do', 'when_to_do']] = final_df.apply(decision_engine, axis=1)
final_df = final_df.drop(columns=['time_period', 'stress_level'])

final_df.to_csv('predictions.csv', index=False)
print("\nFinal 'predictions.csv' generated for the Test Set!")