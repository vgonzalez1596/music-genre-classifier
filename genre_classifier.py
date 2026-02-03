# Spotify Genre Classifier (scikit-learn)
# Trains a classifier to predict song genre from audio features using the Kaggle Spotify Tracks dataset.

# Outputs:
# - results/metrics.txt
# - results/confusion_matrix.png
# - results/feature_importance.png

# Importing necessary packages/functions
from pathlib import Path
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# Defining what features and genres we want to use in the model training
FEATURE_COLS = [
    "popularity",
    "danceability",
    "energy",
    "key",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "tempo",
]

DESIRED_GENRES = ["acoustic", "country", "hip-hop", "metal", "tango"]

# Removing 5 specific tracks so they are never used in training; they will be used for testing the model later
TRACK_IDS_TO_REMOVE = [
    "1EzrEOXmMH3G43AXT1y7pA",
    "2SpEHTbUuebeLkgs9QB7Ue",
    "7KXjTSCq5nL1LoYtL7XAwS",
    "2MuWTIM3b0YEAskbeeFE1i",
    "3VTQ6Fl09GsFonGJiukMTE",
]

# Create output folder
def ensure_dirs():
    Path("results").mkdir(exist_ok=True)

# Load in dataset and then clean/filter it
def load_and_clean(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    required_cols = set(FEATURE_COLS + ["track_genre"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    has_track_id = "track_id" in df.columns

    # Filter to genres of interest
    df = df[df["track_genre"].isin(DESIRED_GENRES)].copy()

    # Drop NA rows for relevant columns
    needed = FEATURE_COLS + ["track_genre"]
    if has_track_id:
        needed += ["track_id"]
    df = df.dropna(subset=needed).copy()

    # Remove specified tracks
    if has_track_id:
        df = df[~df["track_id"].isin(TRACK_IDS_TO_REMOVE)].copy()

    # Converting the genre column from strings to integers to be able to use the machine learning algorithm
    # 0=acoustic, 1=country, 2=hip-hop, 3=metal, 4=tango
    df["genre_code"] = pd.Categorical(df["track_genre"], categories=DESIRED_GENRES).codes

    return df

#Model training 
def train_random_forest(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df[FEATURE_COLS].copy()
    y = df["genre_code"].copy()

    # Scaling the features to ensure all features are along the same scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Initializing the Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )

    # Training the model
    model.fit(X_train, y_train)

    # Making predictions on the test data and evaluating the model's accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, target_names=DESIRED_GENRES)

    return model, scaler, acc, report, (y_test, y_pred)

def save_results(model, acc: float, report: str, y_test, y_pred):
    # Save metrics
    with open("results/metrics.txt", "w") as f:
        f.write(f"Random Forest Accuracy: {acc:.2f}\n\n")
        f.write(report)

    # Confusion matrix (shows predicted vs true labels)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=DESIRED_GENRES,
        xticks_rotation=45,
    )
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=200)
    plt.close()

    # Feature importance
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]

    plt.figure()
    plt.bar([FEATURE_COLS[i] for i in order], importances[order])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=200)
    plt.close()

# Inputting 5 songs (previously removed from dataset and not trained on) and determining their predicted genre
def predict_examples(model, scaler):
    examples = {
        "I'm Yours (Jason Mraz)": [80, 0.703, 0.444, 11, -9.331, 0.0417, 0.559, 0, 150.96],
        "Jolene (Dolly Parton)": [73, 0.674, 0.537, 1, -10.971, 0.0363, 0.566, 0, 110.578],
        "HUMBLE. (Kendrick Lamar)": [84, 0.908, 0.621, 1, -6.638, 0.102, 0.000282, 0.0000539, 150.011],
        "Master of Puppets (Metallica)": [81, 0.543, 0.836, 4, -9.11, 0.0353, 0.000647, 0.431, 105.173],
        "La Cumparsita (Carlos Gardel)": [16, 0.687, 0.46, 8, -6.536, 0.0785, 0.985, 0, 137.994],
    }

    print("\nExample predictions:")
    for name, metrics in examples.items():
        x = pd.DataFrame([metrics], columns=FEATURE_COLS)
        x_scaled = scaler.transform(x)
        pred_code = model.predict(x_scaled)[0]
        print(f"- {name} → {DESIRED_GENRES[pred_code]}")

def main():
    parser = argparse.ArgumentParser(description="Train Spotify genre classifier.")
    parser.add_argument("--data_path", default="data/Spotify_Dataset.csv", help="Path to Spotify_Dataset.csv")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    ensure_dirs()

    df = load_and_clean(args.data_path)
    if len(df) < 200:
        raise ValueError("Filtered dataset is very small. Check that genre names match your dataset.")

    model, scaler, acc, report, (y_test, y_pred) = train_random_forest(
        df, test_size=args.test_size, random_state=args.random_state
    )

    save_results(model, acc, report, y_test, y_pred)

    print(f"Random Forest Accuracy: {acc:.2f}")
    print("Saved: results/metrics.txt, results/confusion_matrix.png, results/feature_importance.png")

    predict_examples(model, scaler)


if __name__ == "__main__":
    main()
