# model.py

# Import the load_imports function from imports.py
from Imports import load_imports

# Load all necessary imports
imports = load_imports()

# Assign the imported modules to local variables
GaussianNB = imports['GaussianNB']
GradientBoostingClassifier = imports['GradientBoostingClassifier']
RandomForestClassifier = imports['RandomForestClassifier']
KNeighborsClassifier = imports['KNeighborsClassifier']
accuracy_score = imports['accuracy_score']
classification_report = imports['classification_report']
confusion_matrix = imports['confusion_matrix']
pd = imports['pd']

class ModelRunner:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train_and_evaluate(self, clf):
        """Train and evaluate the given classifier."""
        clf.fit(self.X_train, self.y_train)
        y_train_pred = clf.predict(self.X_train)
        y_test_pred = clf.predict(self.X_test)
        accuracy_train = accuracy_score(self.y_train, y_train_pred)
        accuracy_test = accuracy_score(self.y_test, y_test_pred)
        report_train = classification_report(self.y_train, y_train_pred)
        report_test = classification_report(self.y_test, y_test_pred)
        cm = confusion_matrix(self.y_test, y_test_pred)
        return accuracy_train, accuracy_test, report_train, report_test, cm

def main():
    try:
        # Load the data from CSV files
        y_train_smote = pd.read_csv('y_train_smote.csv').values.ravel()
        y_test = pd.read_csv('y_test.csv').values.ravel()

        feature_sets = ['rf', 'chi2', 'anova', 'common']
        classifiers = {
            'Naive Bayes': GaussianNB(),
            'GBM': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }

        for feature_set in feature_sets:
            X_train_scaled = pd.read_csv(f'X_train_scaled_{feature_set}.csv')
            X_test_scaled = pd.read_csv(f'X_test_scaled_{feature_set}.csv')

            for clf_name, clf in classifiers.items():
                print(f"\n{clf_name} on {feature_set} feature set")
                runner = ModelRunner(X_train_scaled, y_train_smote, X_test_scaled, y_test)
                accuracy_train, accuracy_test, report_train, report_test, cm = runner.train_and_evaluate(clf)
                print("Training Accuracy:", accuracy_train)
                print("Testing Accuracy:", accuracy_test)
                print("Classification Report (Train):\n", report_train)
                print("Classification Report (Test):\n", report_test)
                print("Confusion Matrix (Test):\n", cm)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
