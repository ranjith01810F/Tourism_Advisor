# feature_selection.py

# Import the load_imports function from imports.py
from Imports import load_imports

# Load all necessary imports
imports = load_imports()
                                   
# Assign the imported modules to local variables
pd = imports['pd']
LabelEncoder = imports['LabelEncoder']
RandomForestClassifier = imports['RandomForestClassifier']
SelectKBest = imports['SelectKBest']
chi2 = imports['chi2']
f_classif = imports['f_classif']
logging = imports['logging']

class FeatureSelection:
    def __init__(self, df):
        self.df = df

    def label_encode(self, columns):
        """Encode categorical columns using Label Encoding."""
        label_encoder = LabelEncoder()
        for col in columns:
            self.df[col] = label_encoder.fit_transform(self.df[col])
        return self.df

    def select_features_rf(self, X, y, n_features=10):
        """Select top features using Random Forest importance."""
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        importances = clf.feature_importances_
        feature_names = X.columns

        feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        top_features_rf = feature_importances.head(n_features)['Feature'].values
        logging.info(f"Top {n_features} features (Random Forest): {top_features_rf}")
        return top_features_rf

    def select_features_chi2(self, X, y, n_features=10):
        """Select top features using chi2."""
        selector = SelectKBest(score_func=chi2, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features_chi2 = X.columns[selected_indices]
        logging.info(f"Selected Features (chi2): {selected_features_chi2}")
        return selected_features_chi2

    def select_features_anova(self, X, y, n_features=10):
        """Select top features using ANOVA."""
        selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features_anova = X.columns[selected_indices]
        logging.info(f"Selected Features (ANOVA): {selected_features_anova}")
        return selected_features_anova

    def find_common_features(self, top_features_rf, selected_features_chi2, selected_features_anova):
        """Find common features selected by all three methods."""
        common_features = list(set(top_features_rf) & set(selected_features_chi2) & set(selected_features_anova))
        logging.info(f"Common features selected by Random Forest, Chi-Square, and ANOVA: {common_features}")
        return common_features

def main():
    try:
        # Import the cleaned dataset
        df = pd.read_csv('cleaned_dataset.csv')

        # Create an instance of FeatureSelection
        fs = FeatureSelection(df)

        # Label Encoding
        columns_to_encode = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']
        df_encoded = fs.label_encode(columns_to_encode)
        logging.info("Label encoding completed")

        # Define the features and target variable
        X = df_encoded.drop(columns=['ProdTaken'])
        y = df_encoded['ProdTaken']

        # Feature Selection
        top_features_rf = fs.select_features_rf(X, y)
        selected_features_chi2 = fs.select_features_chi2(X, y)
        selected_features_anova = fs.select_features_anova(X, y)

        # Find common features
        common_features = fs.find_common_features(top_features_rf, selected_features_chi2, selected_features_anova)
        logging.info(f"Common features: {common_features}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
