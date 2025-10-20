import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Učitava CSV fajl i vraća DataFrame.

    Args:
        file_path (str): Putanja do CSV fajla.

    Returns:
        pd.DataFrame: Učitani podaci.

    Raises:
        FileNotFoundError: Ako fajl ne postoji.
    """
    try:
        df = pd.read_csv(file_path)
        print(df.head())
        print(df.info())
        print(df.describe())
        return df
    except FileNotFoundError:
        print("Fajl nije pronađen. Proveri putanju.")
        raise

if __name__ == "__main__":
    load_data("../data/leak_detection.csv")  # Prilagodi putanju ako treba