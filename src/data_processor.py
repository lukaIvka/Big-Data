import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
import joblib
from typing import Tuple, Optional

# Postavi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        self.scaler = MinMaxScaler()

    def load_data(self) -> pd.DataFrame:
        try:
            logger.info(f"Učitavanje podataka iz {self.file_path}")
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Učitano {len(self.df)} redova sa {len(self.df.columns)} kolona.")
            return self.df
        except FileNotFoundError:
            logger.error(f"Fajl {self.file_path} nije pronađen.")
            raise
        except Exception as e:
            logger.error(f"Greška pri učitavanju podataka: {str(e)}")
            raise

    def clean_data(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Podaci nisu učitani. Pozovi load_data() prvo.")
        
        logger.info("Početak čišćenja podataka.")
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f"Uklonjeno {initial_rows - len(self.df)} duplikata.")

        if self.df.isnull().sum().sum() > 0:
            logger.warning("Pronađene null vrednosti. Popunjavanje sa medianom.")
            self.df = self.df.fillna(self.df.median())
        else:
            logger.info("Nema null vrednosti.")

        return self.df

    def feature_engineering(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Podaci nisu učitani. Pozovi load_data() prvo.")
        
        logger.info("Početak feature engineering-a.")
        self.df['Pressure_Delta'] = self.df['Pressure'].diff().abs()
        if self.df['Pressure_Delta'].isnull().sum() > 0:
            logger.warning("Pronađeni NaN u Pressure_Delta. Popunjavanje sa medianom.")
            self.df['Pressure_Delta'] = self.df['Pressure_Delta'].fillna(self.df['Pressure_Delta'].median())

        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['Hour'] = self.df['timestamp'].dt.hour
        else:
            logger.warning("Nema 'timestamp' kolone za feature engineering.")

        categorical_cols = ['Zone', 'Block', 'Pipe', 'Location_Code']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = pd.Categorical(self.df[col]).codes

        return self.df

    def scale_data(self) -> Tuple[np.ndarray, MinMaxScaler, pd.Series]:
        if self.df is None:
            raise ValueError("Podaci nisu učitani. Pozovi load_data() prvo.")
        
        logger.info("Početak skaliranja podataka.")
        labels = self.df['Leakage_Flag'].copy()
        features = self.df.drop(columns=['Leakage_Flag'])
        
        # Analiza distribucije klasa
        leakage_count = labels.value_counts()
        logger.info(f"Distribucija Leakage_Flag: {leakage_count.to_dict()}")
        
        # Izbaci kolone sa varijansom 0 pre skaliranja
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if features[col].var() == 0:
                logger.warning(f"Kolona {col} ima varijansu 0, uklanjam je.")
                features = features.drop(columns=[col])
        
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("Nema numeričkih kolona za skaliranje nakon uklanjanja konstantnih.")
        
        features_scaled = features.copy()
        features_scaled[numeric_cols] = self.scaler.fit_transform(features[numeric_cols])
        df_scaled = features_scaled[numeric_cols].values
        joblib.dump(self.scaler, '../data/scaler.pkl')
        np.savetxt('../data/scaled_data.csv', df_scaled, delimiter=',')
        return df_scaled, self.scaler, labels

    def get_statistics(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Podaci nisu učitani. Pozovi load_data() prvo.")
        
        logger.info("Izračunavanje statistika.")
        stats = self.df.describe()
        correlation = self.df.corr()['Leakage_Flag'].sort_values(ascending=False)
        return pd.DataFrame({'Statistics': stats.loc['mean'], 'Correlation with Leakage': correlation})

if __name__ == "__main__":
    processor = DataProcessor("../data/location_aware_gis_leakage_dataset.csv")
    df = processor.load_data()
    df_clean = processor.clean_data()
    df_features = processor.feature_engineering()
    df_scaled, scaler, labels = processor.scale_data()
    stats = processor.get_statistics()
    print("Deskriptivne statistike i korelacije:")
    print(stats)