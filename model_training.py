import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib # For saving the model

def train_model(input_csv_path='processed_stock_data.csv', model_output_path='random_forest_quality_model.joblib'):
    """
    Loads processed data, splits it, trains a Random Forest classifier,
    and saves the trained model.
    """
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Loaded processed data from {input_csv_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file {input_csv_path} was not found. Please run data preprocessing first.")
        return None

    if df.empty:
        print(f"Error: The input data file {input_csv_path} is empty.")
        return None

    # --- 1. Define Features (X) and Target (y) ---
    feature_columns = ['ROE', 'DE_Ratio', 'PE_Ratio']
    target_column = 'Quality'

    if not all(col in df.columns for col in feature_columns):
        print(f"Error: Not all feature columns {feature_columns} found in the input data.")
        return None
    if target_column not in df.columns:
        print(f"Error: Target column '{target_column}' not found in the input data.")
        return None

    X = df[feature_columns]
    y = df[target_column]

    print(f"\nFeatures (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    # Check if target variable has more than one class
    if y.nunique() < 2:
        print(f"Error: The target variable '{target_column}' has less than 2 unique classes. Classification requires at least 2 classes.")
        print(f"Value counts for target variable:\n{y.value_counts()}")
        # This can happen if, after preprocessing, all remaining samples belong to one class.
        return None

    # --- 2. Split Data ---
    # Using 80% for training and 20% for testing.
    # random_state ensures reproducibility of the split.
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"\nData split into training and testing sets.")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
        print(f"Testing target distribution:\n{y_test.value_counts(normalize=True)}")

    except ValueError as e:
        # This can happen if test_size results in too few samples for one class for stratification
        print(f"Error during train_test_split (potentially due to small dataset/class imbalance for stratification): {e}")
        print("Attempting split without stratification as a fallback for very small datasets...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Fallback split successful (without stratification).")
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        except Exception as e_fallback:
            print(f"Fallback split also failed: {e_fallback}. Cannot proceed with model training.")
            return None


    # --- 3. Instantiate Model ---
    # Using RandomForestClassifier with a random_state for reproducibility.
    # n_estimators is the number of trees in the forest. Default is 100.
    # Other parameters can be tuned (e.g., max_depth, min_samples_split).
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')
    # Added class_weight='balanced_subsample' as the dataset is small and potentially imbalanced.

    # --- 4. Train Model ---
    try:
        print("\nTraining the Random Forest model...")
        model.fit(X_train, y_train)
        print("Model training completed.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

    # --- 5. Save the Model ---
    try:
        joblib.dump(model, model_output_path)
        print(f"Trained model saved to {model_output_path}")
    except Exception as e:
        print(f"Error saving the model: {e}")
        # Don't necessarily return None here, as model might be trained but just not saved.
        # However, for the workflow, saving is important.

    # Also save X_test and y_test for easy loading in the evaluation script
    try:
        X_test.to_csv('X_test.csv', index=False)
        y_test.to_csv('y_test.csv', index=False, header=True) # Save y_test with header as it's a Series
        print("X_test and y_test saved to CSV files for evaluation.")
    except Exception as e:
        print(f"Error saving X_test/y_test: {e}")

    return model


if __name__ == '__main__':
    print("Starting model training script...")
    # The train_model function will load data, train, and save the model.
    # It also saves X_test and y_test for the evaluation step.
    trained_model = train_model()

    if trained_model:
        print("\nModel training script finished successfully.")
    else:
        print("\nModel training script failed or did not complete.")
