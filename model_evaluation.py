import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path='random_forest_quality_model.joblib',
                   x_test_path='X_test.csv',
                   y_test_path='y_test.csv'):
    """
    Loads a trained model and test data, makes predictions,
    and prints evaluation metrics.
    """
    # --- 1. Load Model and Test Data ---
    try:
        model = joblib.load(model_path)
        print(f"Loaded trained model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        X_test = pd.read_csv(x_test_path)
        print(f"Loaded X_test data from {x_test_path}. Shape: {X_test.shape}")
    except FileNotFoundError:
        print(f"Error: X_test file {x_test_path} not found.")
        return
    except Exception as e:
        print(f"Error loading X_test: {e}")
        return

    try:
        y_test_df = pd.read_csv(y_test_path)
        # Assuming y_test was saved as a DataFrame with the target column name
        # If it was saved as a Series without a header, it might load with a default column name like '0'
        # Or, if it was saved with a specific name (e.g., 'Quality'), use that.
        # Let's assume the column is named 'Quality' as in the original dataframe or it's the first column.
        if 'Quality' in y_test_df.columns:
            y_test = y_test_df['Quality']
        elif not y_test_df.empty and len(y_test_df.columns) == 1: # If only one column, assume it's the target
            y_test = y_test_df.iloc[:, 0]
        else:
            print(f"Error: Could not determine target column in {y_test_path}. Expected 'Quality' or a single column.")
            return
        print(f"Loaded y_test data from {y_test_path}. Shape: {y_test.shape}")
    except FileNotFoundError:
        print(f"Error: y_test file {y_test_path} not found.")
        return
    except Exception as e:
        print(f"Error loading y_test: {e}")
        return

    if X_test.empty or y_test.empty:
        print("Error: Test data (X_test or y_test) is empty.")
        return
    if len(X_test) != len(y_test):
        print(f"Error: X_test (rows: {len(X_test)}) and y_test (rows: {len(y_test)}) have mismatched lengths.")
        return

    # --- 2. Make Predictions ---
    try:
        y_pred = model.predict(X_test)
        print("\nMade predictions on the test set.")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return

    # --- 3. Calculate and Display Evaluation Metrics ---
    print("\n--- Model Evaluation Results ---")

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:")
    # Handle potential UndefinedMetricWarning for precision/recall if a class has no predicted samples
    # zero_division=0 means if a class has no predicted samples, precision/recall for that class will be 0.
    # zero_division=1 means it will be 1 (not typically desired).
    # 'warn' is the default.
    try:
        class_report = classification_report(y_test, y_pred, zero_division=0)
        print(class_report)
    except Exception as e:
        print(f"Could not generate classification report: {e}")


    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot Confusion Matrix
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Save the plot to a file
        plot_filename = 'confusion_matrix.png'
        plt.savefig(plot_filename)
        print(f"\nConfusion matrix plot saved to {plot_filename}")
        # plt.show() # This might not work in all environments; saving is more robust.
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

    # --- 4. Feature Importances ---
    print("\n--- Feature Importances ---")
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns # Assuming X_test has original feature names as columns

        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

        print(feature_importance_df)

        # Plot Feature Importances
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
            plt.title('Feature Importances')
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')

            plot_filename_fi = 'feature_importances.png'
            plt.tight_layout() # Adjust layout to make sure everything fits
            plt.savefig(plot_filename_fi)
            print(f"\nFeature importances plot saved to {plot_filename_fi}")
            # plt.show()
        except Exception as e:
            print(f"Error plotting feature importances: {e}")
    else:
        print("The model does not have feature_importances_ attribute (e.g., not a tree-based model).")


if __name__ == '__main__':
    print("Starting model evaluation script (including feature importance)...")
    evaluate_model()
    print("\nModel evaluation script finished.")
