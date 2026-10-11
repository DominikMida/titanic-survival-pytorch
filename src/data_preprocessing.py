import numpy as np
import pandas as pd
import pickle
from typing import Tuple


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Data loaded successfully")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Train columns: {list(train_df.columns)}")
    
    return train_df, test_df

def handle_missing_values(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    age_median = train_df['wiek'].median()
    port_mode = train_df['port_wsiadania'].mode()[0]
    fare_median = train_df['oplata'].median()
    
    for df in [train_df, test_df]:
        df['wiek'] = df['wiek'].fillna(age_median)
        df['port_wsiadania'] = df['port_wsiadania'].fillna(port_mode)
        df['oplata'] = df['oplata'].fillna(fare_median)
    
    print("[OK] Missing values handled")
    return train_df, test_df


def handle_outliers(train_df: pd.DataFrame) -> pd.DataFrame:
    q1 = train_df['oplata'].quantile(0.25)
    q3 = train_df['oplata'].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    train_df['oplata'] = train_df['oplata'].clip(upper=upper_bound)
    print(f"[OK] Outliers removed (fare capped at {upper_bound:.2f})")
    
    return train_df


def drop_irrelevant_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    drop_cols = [
        'nr_pasazera', 'sny_przed_podroza', 'ostatni_posilek',
        'wodka_wypita_ml', 'frakcja_majonezowa',
        'cebula_w_bagazu_kg', 'znajomosc_tekstow_disco_polo'
    ]
    
    train_df = train_df.drop(columns=drop_cols, errors='ignore')
    test_df = test_df.drop(columns=drop_cols, errors='ignore')
    
    print(f"[OK] Dropped {len(drop_cols)} irrelevant columns")
    return train_df, test_df


def extract_titles(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def extract_tytul(name):
        return name.strip().split()[0]
    
    for df in [train_df, test_df]:
        df['tytul'] = df['pasazer'].apply(extract_tytul)
    
    plec_map = {
        'Pan': 0, 'Panicz': 0, 'Ks.': 0, 'Dr': 0, 'Dr.': 0,
        'Mjr': 0, 'Mjr.': 0, 'Płk': 0, 'Kpt.': 0, 'Hr.': 0,
        'Prof.': 0, 'Prof': 0,
        'Pani': 1, 'Panna': 1,
    }
    
    for df in [train_df, test_df]:
        df['plec'] = df['tytul'].map(plec_map).fillna(-1).astype(int)
    
    train_df = train_df.drop(columns=['pasazer'])
    test_df = test_df.drop(columns=['pasazer'])
    
    print("[OK] Titles extracted and mapped to gender")
    return train_df, test_df


def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    num_cols = ['wiek', 'rodzenstwo_malzonek', 'rodzice_dzieci', 'oplata']
    
    num_stats = {}
    for col in num_cols:
        mean = train_df[col].mean()
        std = train_df[col].std()
        num_stats[col] = (mean, std)
        
        train_df[col] = (train_df[col] - mean) / std
        test_df[col] = (test_df[col] - mean) / std
    
    print("[OK] Numerical features normalized (z-score)")
    return train_df, test_df, num_stats


def encode_categorical_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    categorical_cols = ['port_wsiadania', 'strona_burty', 'tytul']

    train_df = pd.get_dummies(train_df, columns=categorical_cols, prefix=categorical_cols)
    test_df = pd.get_dummies(test_df, columns=categorical_cols, prefix=categorical_cols)
    
    train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

    train_df = train_df.astype({col: int for col in train_df.select_dtypes('bool').columns})
    test_df = test_df.astype({col: int for col in test_df.select_dtypes('bool').columns})
    
    print("[OK] Categorical features one-hot encoded")
    return train_df, test_df


def prepare_training_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_train = train_df.drop(columns=['przezycie']).values.astype(np.float32)
    y_train = train_df['przezycie'].values.astype(np.int64)
    X_test = test_df.drop(columns=['przezycie'], errors='ignore').values.astype(np.float32)
    
    print(f"[OK] Training data prepared: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[OK] Test data prepared: X_test={X_test.shape}")
    
    return X_train, y_train, X_test


def preprocess_pipeline(train_path: str, test_path: str, 
                       output_train: str = "data/processed/wisla_train.pkl",
                       output_test: str = "data/processed/wisla_test.pkl") -> None:
    print("\n" + "="*50)
    print("TITANIC DATA PREPROCESSING PIPELINE")
    print("="*50 + "\n")
    
    train_df, test_df = load_data(train_path, test_path)
    print(f"Data loaded: train={train_df.shape}, test={test_df.shape}\n")
    
    train_df, test_df = handle_missing_values(train_df, test_df)
    train_df = handle_outliers(train_df)
    train_df, test_df = drop_irrelevant_columns(train_df, test_df)
    train_df, test_df = extract_titles(train_df, test_df)
    train_df, test_df, _ = normalize_features(train_df, test_df)
    train_df, test_df = encode_categorical_features(train_df, test_df)
    
    X_train, y_train, X_test = prepare_training_data(train_df, test_df)

    import os
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    
    with open(output_train, 'wb') as f:
        pickle.dump((X_train, y_train), f)
    with open(output_test, 'wb') as f:
        pickle.dump(X_test, f)
    
    print(f"\n[OK] Preprocessed data saved to:")
    print(f"  - {output_train}")
    print(f"  - {output_test}\n")


if __name__ == "__main__":
    preprocess_pipeline(
        train_path="data/raw/wisla_train.csv", 
        test_path="data/raw/wisla_test.csv"
    )