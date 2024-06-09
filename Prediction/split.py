# splitting.py

# Import the load_imports function from imports.py
from Imports import load_imports

# Load all necessary imports
imports = load_imports()

# Assign the imported modules to local variables
train_test_split = imports['train_test_split']
SMOTE = imports['SMOTE']
StandardScaler = imports['StandardScaler']
logging = imports['logging']
pd = imports['pd']

class DataProcessing:
    def __init__(self, df, columns_to_encode):
        from Feature_selection import FeatureSelection
        self.df = df
        self.columns_to_encode = columns_to_encode
        self.fs = FeatureSelection(df)
        self.df_encoded = None
        self.X = None
        self.y = None

    def label_encode(self):
        self.df_encoded = self.fs.label_encode(self.columns_to_encode)
        logging.info("Label encoding completed")
        return self.df_encoded

    def define_features_target(self):
        self.X = self.df_encoded.drop(columns=['ProdTaken'])
        self.y = self.df_encoded['ProdTaken']
        return self.X, self.y

    def feature_selection(self, n_features=10):
        top_features_rf = list(self.fs.select_features_rf(self.X, self.y, n_features))
        selected_features_chi2 = list(self.fs.select_features_chi2(self.X, self.y, n_features))
        selected_features_anova = list(self.fs.select_features_anova(self.X, self.y, n_features))
        common_features = list(self.fs.find_common_features(top_features_rf, selected_features_chi2, selected_features_anova))
        return top_features_rf, selected_features_chi2, selected_features_anova, common_features

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logging.info(f"Shape of X_train: {X_train.shape}")
    logging.info(f"Shape of X_test: {X_test.shape}")
    logging.info(f"Shape of y_train: {y_train.shape}")
    logging.info(f"Shape of y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to the training data."""
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    logging.info(f"Shape of X_train_smote: {X_train_smote.shape}")
    logging.info(f"Shape of y_train_smote: {y_train_smote.shape}")
    logging.info(f"Value counts for y_train_smote: {y_train_smote.value_counts()}")
    return X_train_smote, y_train_smote

def scale_data(X_train, X_test):
    """Scale the training and testing data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info(f"Shape of X_train_scaled: {X_train_scaled.shape}")
    logging.info(f"Shape of X_test_scaled: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled

def save_splits(X_train_scaled, X_test_scaled, y_train_smote, y_test, feature_set_name):
    """Save the splits to CSV files."""
    pd.DataFrame(X_train_scaled).to_csv(f'X_train_scaled_{feature_set_name}.csv', index=False)
    pd.DataFrame(X_test_scaled).to_csv(f'X_test_scaled_{feature_set_name}.csv', index=False)
    pd.DataFrame(y_train_smote).to_csv('y_train_smote.csv', index=False)
    pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

def process_and_save_splits(X, y, feature_set_name):
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Apply SMOTE
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Scale the data
    X_train_scaled, X_test_scaled = scale_data(X_train_smote, X_test)

    # Save the splits
    save_splits(X_train_scaled, X_test_scaled, y_train_smote, y_test, feature_set_name)

    # Print the splits information
    logging.info(f"Shape of X_train_scaled_{feature_set_name}: {X_train_scaled.shape}")
    logging.info(f"Shape of X_test_scaled_{feature_set_name}: {X_test_scaled.shape}")
    logging.info(f"Value counts for y_train_smote_{feature_set_name}: {pd.Series(y_train_smote).value_counts()}")

def main():
    try:
        # Import the cleaned dataset
        cleaned_df = pd.read_csv('cleaned_dataset.csv')

        # Define columns to encode
        columns_to_encode = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
        
        # Data processing and feature selection
        dp = DataProcessing(cleaned_df, columns_to_encode)
        
        # Label Encoding
        df_encoded = dp.label_encode()
        
        # Define the features and target variable
        X, y = dp.define_features_target()

        # Feature Selection
        top_features_rf, selected_features_chi2, selected_features_anova, common_features = dp.feature_selection()

        # Process and save splits for each feature set
        feature_sets = {
            'rf': top_features_rf,
            'chi2': selected_features_chi2,
            'anova': selected_features_anova,
            'common': common_features
        }

        for feature_set_name, feature_set in feature_sets.items():
            if feature_set:
                X_selected = X[feature_set]
                process_and_save_splits(X_selected, y, feature_set_name)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
