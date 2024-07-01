# %% [markdown]
# # PHASE - I
# ## 1.1 Data Ingestion:
#     -The dataset is collected from Kaggle.
#     -In this phase, we will load the dataset into a pandas DataFrame to facilitate further analysis. The dataset will be inspected to ensure it has been successfully loaded.

# %%
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("Travel.csv")
df

#comment


# %% [markdown]
# ## PHASE - II
# ### 2.1 Data Understanding:
# 
# In this phase, we aim to gain a comprehensive understanding of the dataset. This involves examining the structure, summary statistics, and distributions of the data. Key steps include:
# 
# 1. **Data Overview**: Reviewing the dataset's columns and data types to understand what kind of information is available.
# 
#    - Examine the first few rows of the dataset to understand its structure and contents.
#    - Check the data types and missing values to assess data quality and completeness.
# 
# 2. **Descriptive Statistics**: Calculating summary statistics (e.g., mean, median, standard deviation) to get a sense of the central tendency and variability of the data.
# 
#    - Compute summary statistics for numerical variables to understand their distribution and variability.
# 
# 3. **Data Distributions**: Visualizing the distributions of key features to identify patterns, outliers, and potential data quality issues.
# 
#    - Generate histograms, box plots, or density plots to visualize the distribution of important variables.
#    - Look for any skewness, multimodality, or outliers in the data distributions.
# 
# 4. **Correlation Analysis**: Analyzing correlations between features to identify relationships that might be important for modeling.
# 
#    - Calculate the correlation matrix between numerical variables to identify strong correlations.
#    - Visualize the correlation matrix using a heatmap to identify patterns of association between variables.
# 

# %%
print(df.columns)


# %% [markdown]
# #### 2.2 Description of Data:
# 
# 1. **CustomerID**: Unique identifier for each customer.
# 2. **ProdTaken**: Indicates whether the customer has taken a product or package (1 for yes, 0 for no).
# 3. **Age**: Age of the customer.
# 4. **TypeofContact**: Mode of contact with the customer (e.g., phone, email).
# 5. **CityTier**: Classification of the city where the customer resides (e.g., Tier 1, Tier 2, Tier 3).
# 6. **DurationOfPitch**: Duration of the sales pitch given to the customer (in minutes).
# 7. **Occupation**: Occupation of the customer.
# 8. **Gender**: Gender of the customer.
# 9. **NumberOfPersonVisiting**: Number of people visiting with the customer.
# 10. **NumberOfFollowups**: Number of follow-ups made to the customer.
# 11. **ProductPitched**: The specific product or package pitched to the customer.
# 12. **PreferredPropertyStar**: Star rating of the preferred property (e.g., 3-star, 4-star, 5-star).
# 13. **MaritalStatus**: Marital status of the customer (e.g., single, married).
# 14. **NumberOfTrips**: Number of trips taken by the customer.
# 15. **Passport**: Indicates whether the customer has a passport (1 for yes, 0 for no).
# 16. **PitchSatisfactionScore**: Customer's satisfaction score for the sales pitch (typically on a scale).
# 17. **OwnCar**: Indicates whether the customer owns a car (1 for yes, 0 for no).
# 18. **NumberOfChildrenVisiting**: Number of children accompanying the customer during the visit.
# 19. **Designation**: Job title or designation of the customer.
# 20. **MonthlyIncome**: Monthly incllness Tourism Package.

# %%
df.info()

# %%
# Calculate and display descriptive statistics for numerical columns
print(df.describe())

# %%
# Calculate and display descriptive statistics for categorical columns
print(df.describe(include=['object']))

# %%
# Check for missing values
missing_values = df.isna().sum()

# Display missing values count
print("Missing Values:")
print(missing_values)


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set(style="whitegrid")

# Function to create bar plots for categorical variables
def plot_bar(column_name):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=column_name, palette='coolwarm')
    plt.title(f'Distribution of {column_name}')
    plt.xticks(rotation=45)
    plt.show()

# Function to create histograms for numerical variables
def plot_hist(column_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column_name, kde=True, color='blue', bins=30)
    plt.title(f'Distribution of {column_name}')
    plt.show()

# Function to create box plots for numerical variables
def plot_box(column_name):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=column_name, palette='coolwarm')
    plt.title(f'Box plot of {column_name}')
    plt.show()

# Plotting the columns as per your requirements
# Bar plots for categorical variables
plot_bar('ProdTaken')
plot_bar('TypeofContact')
plot_bar('CityTier')
plot_bar('Occupation')
plot_bar('Gender')
plot_bar('ProductPitched')
plot_bar('PreferredPropertyStar')
plot_bar('MaritalStatus')
plot_bar('Passport')
plot_bar('OwnCar')
plot_bar('Designation')

# Histograms for numerical variables
plot_hist('Age')
plot_hist('DurationOfPitch')
plot_hist('MonthlyIncome')

# Box plot for MonthlyIncome to see the distribution with outliers
plot_box('MonthlyIncome')


# %%
import random
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
import scipy  # Ensure scipy is imported

# Select only numerical columns for correlation matrix
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr_matrix = df[numerical_cols].corr()


# Count the occurrences of each ProdTaken value
prod_taken_counts = df['ProdTaken'].value_counts().reset_index()
prod_taken_counts.columns = ['ProdTaken', 'Count']

# Create a bar plot for ProdTaken distribution
fig1 = px.bar(prod_taken_counts, x='ProdTaken', y='Count', title='Distribution of ProdTaken')
fig1.show()



# KDE plot for Age distribution, excluding NaN values
fig2 = ff.create_distplot([df['Age'].dropna()], ['Age'], show_hist=False, colors=['blue'])
fig2.update_layout(title='Age Distribution (KDE)')
fig2.show()

# Histogram for Monthly Income distribution
fig3 = px.histogram(df, x='MonthlyIncome', nbins=20, title='Monthly Income Distribution', marginal='box')
fig3.show()

# Count the occurrences of each CityTier
city_tier_counts = df['CityTier'].value_counts().reset_index()
city_tier_counts.columns = ['CityTier', 'Count']

# Count the occurrences of each CityTier value
city_tier_counts = df['CityTier'].value_counts().reset_index()
city_tier_counts.columns = ['CityTier', 'Count']

# Create a bar plot for CityTier distribution with specified colors
fig4 = px.bar(city_tier_counts, x='CityTier', y='Count', title='City Tier Distribution',
              color='CityTier', color_discrete_map={1: 'blue', 2: 'green', 3: 'red'})

fig4.update_layout(xaxis=dict(tickmode='linear'))
fig4.show()
# Pie chart for Gender distribution
fig5 = px.pie(df, names='Gender', title='Gender Distribution', hole=0.3)
fig5.show()

# Count the occurrences of each Occupation value
occupation_counts = df['Occupation'].value_counts().reset_index()
occupation_counts.columns = ['Occupation', 'Count']

# Create a bar plot for Occupation distribution with specified color
fig6 = px.bar(occupation_counts, x='Occupation', y='Count', title='Occupation Distribution',
              color='Occupation', color_discrete_sequence=px.colors.qualitative.Vivid)

fig6.update_layout(showlegend=False)  # Hide the legend if you only want a single color
fig6.show()

# Scatter plot for Age vs Monthly Income
fig7 = px.scatter(df, x='Age', y='MonthlyIncome', color='ProdTaken', title='Age vs Monthly Income')
fig7.show()

# Heatmap of correlations between numerical features
fig8 = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='Magma'
))
fig8.update_layout(title='Correlation Heatmap')
fig8.show()

# Violin plot for Age distribution by ProdTaken
fig9 = px.violin(df, y='Age', color='ProdTaken', box=True, points="all", title='Age Distribution by ProdTaken')
fig9.show()

# Box plot for Monthly Income by Gender
fig10 = px.box(df, x='Gender', y='MonthlyIncome', color='Gender', title='Monthly Income by Gender')
fig10.show()

product_names = df['ProductPitched'].tolist()

# Count the occurrences of each product
product_counts = {product: product_names.count(product) for product in set(product_names)}

# Generate a list of unique colors
unique_colors = []
for _ in range(len(product_counts)):
    unique_colors.append('#{:06x}'.format(random.randint(0, 256**3-1)))

# Create a bar plot
fig11 = go.Figure(go.Bar(
            x=list(product_counts.keys()),
            y=list(product_counts.values()),
            marker_color=unique_colors  # Assign unique colors to each product
))

# Customize the layout
fig11.update_layout(
    title="Product Distribution",
    xaxis_title="Product Names",
    yaxis_title="Frequency"
)

# Show the plot
fig11.show()


# %% [markdown]
# ### Exploratory Data Analysis

# %%
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

# %%
import plotly.express as px
import plotly.subplots as sp
import pandas as pd


# Convert categorical features to string
categorical_features = [
    'ProdTaken', 'TypeofContact', 'CityTier', 'Occupation', 'Gender', 
    'NumberOfPersonVisiting', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'ProductPitched', 
    'MaritalStatus', 'Passport', 'OwnCar', 'Designation'
]

for feature in categorical_features:
    df[feature] = df[feature].astype(str)

# Create the first figure with 8 subplots
fig1 = sp.make_subplots(rows=4, cols=2, subplot_titles=categorical_features[:8])

for i, feature in enumerate(categorical_features[:8]):
    row = i // 2 + 1
    col = i % 2 + 1
    fig1.add_trace(px.histogram(df, x=feature).data[0], row=row, col=col)

fig1.update_layout(height=800, width=1200, title_text="Count Plots of Categorical Features (1/2)")
fig1.show()

# Create the second figure with the remaining 8 subplots
fig2 = sp.make_subplots(rows=4, cols=2, subplot_titles=categorical_features[8:])

for i, feature in enumerate(categorical_features[8:]):
    row = i // 2 + 1
    col = i % 2 + 1
    fig2.add_trace(px.histogram(df, x=feature).data[0], row=row, col=col)

fig2.update_layout(height=800, width=1200, title_text="Count Plots of Categorical Features (2/2)")
fig2.show()


# %%
import plotly.express as px

# Create a sunburst chart
fig = px.sunburst(df, path=['TypeofContact', 'CityTier', 'Occupation', 'ProductPitched'],
                  values='MonthlyIncome', color='ProdTaken', 
                  color_continuous_scale='RdBu',
                  title='Sunburst Chart of Categorical Features and Monthly Income')
fig.update_layout(width=800, height=800)
fig.show()


# %%
import plotly.express as px

# Create violin plots for Monthly Income by ProdTaken
fig = px.violin(df, y='MonthlyIncome', x='ProdTaken', box=True, points='all',
                title='Violin Plot of Monthly Income by Product Taken')
fig.show()

# Create violin plots for Age by ProdTaken
fig = px.violin(df, y='Age', x='ProdTaken', box=True, points='all',
                title='Violin Plot of Age by Product Taken')
fig.show()


# %%
# Example: Stacked bar chart of 'ProdTaken' by 'Gender'
cross_tab = pd.crosstab(df['ProdTaken'], df['Gender'])
cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stacked Bar Chart of ProdTaken by Gender')
plt.xlabel('ProdTaken')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()


# %%
import seaborn as sns

# Plot distribution plots for numerical features
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
numerical_features = ['Age', 'DurationOfPitch',  'MonthlyIncome']

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=feature, kde=True)
    plt.title(f'Distribution plot of {feature}')
    
plt.tight_layout()
plt.show()


# %%
# Example: Faceted histograms of 'Age' by 'ProdTaken'
g = sns.FacetGrid(df, col='ProdTaken', height=5)
g.map_dataframe(sns.histplot, x='Age', bins=20, kde=True)
g.set_titles('Histogram of Age by ProdTaken: {col_name}')
plt.show()


# %%
import matplotlib.pyplot as plt

# Calculate the number of customers who took the product
num_taken = df['ProdTaken'].sum()

# Calculate the number of customers who did not take the product
num_not_taken = len(df) - num_taken

# Plot the conversion rate using a pie chart
labels = ['Taken', 'Not Taken']
sizes = [num_taken, num_not_taken]
colors = ['lightgreen', 'lightcoral']
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Conversion Rate')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# %%
# Calculate the total number of customers
total_customers = len(df)

# Calculate the conversion rate
conversion_rate = num_taken / total_customers * 100

# Print the conversion rate
print("Conversion Rate: {:.2f}%".format(conversion_rate))


# %%
import matplotlib.pyplot as plt

# Calculate conversion rate for each product pitched
conversion_rate_by_product = df.groupby('ProductPitched')['ProdTaken'].mean()

# Plot conversion rate for each product pitched
plt.figure(figsize=(10, 6))
conversion_rate_by_product.plot(kind='bar', color='skyblue')

# Annotate each bar with its value
for i, rate in enumerate(conversion_rate_by_product):
    plt.text(i, rate, '{:.2f}%'.format(rate*100), ha='center', va='bottom')

# Set plot labels and title
plt.title('Conversion Rate by Product Pitched')
plt.xlabel('Product Pitched')
plt.ylabel('Conversion Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show plot
plt.show()


# %% [markdown]
# ## Phase - III- Data Preparation & Data Preprocessing
# 
# **3.1 Handle missing values**
# 

# %% [markdown]
# **3.2 Data Transformation**

# %%
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Assuming df, num_cols, cat_cols, and imputer are already defined
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns
imputer = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), num_cols),
        ('cat', SimpleImputer(strategy='most_frequent'), cat_cols)
    ]
)

imputed_data = imputer.fit_transform(df)

# Convert imputed_data to DataFrame
df = pd.DataFrame(imputed_data, columns=num_cols.tolist() + cat_cols.tolist())

# Check for missing values
print("After handling missing values:\n",df.isnull().sum())

# Replace 'Fe Male' with 'Female'
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')



# %% [markdown]
# 
# 
# **3.2 Data Transformation**

# %%

from sklearn.preprocessing import LabelEncoder

# Select categorical columns
cat_cols = ['TypeofContact','ProdTaken','Occupation', 'Gender','ProductPitched','MaritalStatus', 'Designation']

# Initialize LabelEncoder
encoder = LabelEncoder()

# Encode categorical columns
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# Encode categorical columns and print value counts
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])
    print(f"Value counts for {col}:")
    print(df[col].value_counts())
    print("\n")


# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

feature=df.columns

# Compute cosine similarity between users
user_similarity = cosine_similarity(df[feature])

# Get the index of users who haven't taken any product
non_purchasing_users_idx = df[df['ProdTaken'] == 0].index

# Recommend products
recommendations = {}
for user_idx in non_purchasing_users_idx:
    # Get similar users who have taken a product
    similar_users = np.argsort(-user_similarity[user_idx])
    similar_users = [idx for idx in similar_users if df.iloc[idx]['ProdTaken'] == 1]

    # Recommend the product that the most similar user has taken
    if similar_users:
        recommended_product = df.iloc[similar_users[0]]['ProductPitched']
        recommendations[df.iloc[user_idx]['CustomerID']] = recommended_product
    else:
        recommendations[df.iloc[user_idx]['CustomerID']] = None

# %%
# Decode the product labels back to original
product_labels = {0: 'Basic', 1: 'Deluxe', 2: 'King', 3: 'Standard', 4: 'Super Deluxe'}

# Create a DataFrame for the recommendations
recommendation_df = pd.DataFrame(list(recommendations.items()), columns=['CustomerID', 'RecommendedProduct'])

# Map the numeric product labels to the original class labels
recommendation_df['RecommendedProduct'] = recommendation_df['RecommendedProduct'].map(lambda x: product_labels.get(x, None))

# Save recommendations to a CSV file
recommendation_df.to_csv('recommendations_collaberative.csv', index=False)

# Print the recommendations
print(recommendation_df)


# %%
# Count the recommended products
product_counts = recommendation_df['RecommendedProduct'].value_counts()

# Print the counts of recommended products
print(product_counts)
# Calculate and print the total count of recommended products
total_count = product_counts.sum()
print(f"\nTotal count of recommended products: {total_count}")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Horizontal Bar Plot for recommended products
plt.figure(figsize=(10, 6))
sns.barplot(x=product_counts.values, y=product_counts.index, palette='viridis')
plt.title('Counts of Recommended Products')
plt.xlabel('Count')
plt.ylabel('Product')
plt.show()



# %%
import numpy as np
from scipy import stats

# Generate simulated data for A/B testing of recommended products
control_group = np.random.choice(recommendation_df['RecommendedProduct'].values, size=1000, replace=True)  # Control group
experimental_group = np.random.choice(recommendation_df['RecommendedProduct'].values, size=1000, replace=True)  # Experimental group

# Get unique recommended products
recommended_products = np.unique(recommendation_df['RecommendedProduct'].values)

# Analyze the recommended products' effectiveness through a riveting A/B test
control_product_counts = {product: np.sum(control_group == product) for product in recommended_products}
experimental_product_counts = {product: np.sum(experimental_group == product) for product in recommended_products}

# Calculate the total counts of recommended products in each group
total_control_count = sum(control_product_counts.values())
total_experimental_count = sum(experimental_product_counts.values())

# Perform a robust statistical evaluation using the astounding t-test
t_stat, p_value = stats.ttest_ind(list(control_product_counts.values()), list(experimental_product_counts.values()))

# Print the exhilarating results of the A/B test
print("Control Product Counts:", control_product_counts)
print("Experimental Product Counts:", experimental_product_counts)
print("Total Control Count:", total_control_count)
print("Total Experimental Count:", total_experimental_count)
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Offer heart-pounding insights based on the thrilling statistical analysis
if p_value < 0.05:
    print("The difference in recommended products' effectiveness is statistically significant!")
else:
    print("The difference in recommended products' effectiveness is not statistically significant.")



