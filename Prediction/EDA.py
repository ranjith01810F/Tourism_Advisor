# eda.py

# Import the load_imports function from imports.py
from Imports import load_imports

# Load all necessary imports
imports = load_imports()

# Assign the imported modules to local variables
pd = imports['pd']
px = imports['px']
go = imports['go']
logging = imports['logging']

def perform_eda(df):
    try:
        logging.info("Starting EDA")

        # Distribution of Age
        fig1 = px.histogram(df, x='Age', nbins=10, title='Distribution of Age', marginal='box', hover_data=df.columns)
        fig1.show()

        # Pie Chart of Product Taken
        fig2 = px.pie(df, names='ProdTaken', title='Distribution of Product Taken')
        fig2.show()

        # Boxplot of Monthly Income by Gender
        fig3 = px.box(df, x='Gender', y='MonthlyIncome', title='Monthly Income by Gender', points='all')
        fig3.show()

        # Barplot of Product Pitched by Number of Followups
        fig4 = px.bar(df, x='NumberOfFollowups', y='ProdTaken', color='ProductPitched', barmode='group',
                      title='Product Pitched by Number of Followups')
        fig4.show()

        # Heatmap of correlation matrix
        numeric_columns = df.select_dtypes(include=['number']).columns
        correlation_matrix = df[numeric_columns].corr()

        fig5 = go.Figure(data=go.Heatmap(
                           z=correlation_matrix.values,
                           x=correlation_matrix.columns,
                           y=correlation_matrix.columns,
                           colorscale='Viridis'))

        fig5.update_layout(title='Correlation Matrix')
        fig5.show()

        # Scatter Plot of Age vs Monthly Income by Gender
        fig6 = px.scatter(df, x='Age', y='MonthlyIncome', color='Gender',
                          title='Age vs Monthly Income by Gender',
                          hover_data=['Designation', 'CityTier'])
        fig6.show()

        # Pie Chart of Marital Status Distribution
        fig7 = px.pie(df, names='MaritalStatus', title='Marital Status Distribution')
        fig7.show()

        # Sunburst Chart for Multi-level Hierarchical Data (CityTier -> Gender -> ProductPitched)
        fig8 = px.sunburst(df, path=['CityTier', 'Gender', 'ProductPitched'],
                           values='CustomerID', title='Sunburst Chart of CityTier, Gender and Product Pitched')
        fig8.show()

        # Violin Plot of Monthly Income by Designation
        fig9 = px.violin(df, x='Designation', y='MonthlyIncome',
                         title='Monthly Income by Designation',
                         box=True, points='all')
        fig9.show()

        # Grouped Bar Plot for Number of Trips vs Age by Product Taken
        fig10 = px.bar(df, x='Age', y='NumberOfTrips', color='ProdTaken',
                        title='Number of Trips vs Age by Product Taken',
                        barmode='group', hover_data=['CustomerID'])
        fig10.show()

        # Scatter Plot of Age vs Duration of Pitch colored by Gender
        fig11 = px.scatter(df, x='Age', y='DurationOfPitch', color='Gender',
                           title='Age vs Duration of Pitch by Gender',
                           hover_data=['CustomerID'])
        fig11.show()

        logging.info("EDA completed successfully")

    except Exception as e:
        logging.error(f"An error occurred during EDA: {e}")

# Example usage
if __name__ == "__main__":
    # Import the cleaned dataset
    cleaned_df = pd.read_csv('cleaned_dataset.csv')

    if cleaned_df is not None:
        perform_eda(cleaned_df)
    else:
        logging.error("Failed to perform EDA due to missing cleaned data")
