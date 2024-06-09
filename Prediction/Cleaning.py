# cleaning.py

# Import the load_imports function from imports.py
from Imports import load_imports

# Load all necessary imports
imports = load_imports()

# Assign the imported modules to local variables
pd = imports['pd']
SimpleImputer = imports['SimpleImputer']

def clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(df.head())

    # Display the general information about the dataset
    print("\nGeneral Information:")
    print(df.info())

    # Descriptive statistics for numerical columns
    numerical_stats = df.describe()

    # Descriptive statistics for categorical columns
    categorical_stats = pd.DataFrame()
    for col in df.select_dtypes(include=['object', 'category']):
        categorical_stats[col] = [df[col].nunique(),
                                  df[col].value_counts().idxmax(),
                                  df[col].value_counts().max() / len(df) * 100]

    categorical_stats = categorical_stats.rename(columns={0: 'Unique Categories', 1: 'Most Common Category', 2: 'Percentage'})

    # Print descriptive statistics
    print("Descriptive Statistics for Numerical Columns:")
    print(numerical_stats)
    print("\nDescriptive Statistics for Categorical Columns:")
    print(categorical_stats)

    # Check data types
    print("\nData Types:")
    print(df.dtypes)

    # Identify missing values
    missing_values = df.isna().sum()
    print("Missing Values:")
    print(missing_values)

    # Replace 'Fe Male' with 'Female'
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

    # Imputation code
    numerical_cols = df.select_dtypes(include='number').columns
    categorical_cols = df.select_dtypes(exclude='number').columns

    # Handling missing values for numerical columns using mean imputation
    imputer_mean = SimpleImputer(strategy='mean')
    df[numerical_cols] = imputer_mean.fit_transform(df[numerical_cols])

    # Handling missing values for categorical columns using mode imputation
    imputer_mode = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = imputer_mode.fit_transform(df[categorical_cols])

    # Check if there are any remaining missing values
    print("Remaining missing values:")
    print(df.isnull().sum())

    # Return the cleaned DataFrame
    return df

# Define the path to your dataset
file_path = r"C:\\Users\\User\\Documents\\GitHub\\Tourism_Advisor\\Travel.csv"

# Clean the dataset and save the cleaned data
cleaned_df = clean_data(file_path)

cleaned_df.to_csv('cleaned_dataset.csv', index=False)
print("Cleaned data saved to 'cleaned_dataset.csv'")
