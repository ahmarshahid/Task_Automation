
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def clean_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    # For simplicity, we'll fill numerical columns with the mean and categorical columns with the mode
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].mean(), inplace=True)

    # Normalize numerical columns
    scaler = StandardScaler()
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column] = scaler.fit_transform(df[[column]])

    # Encode categorical variables
    encoder = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = encoder.fit_transform(df[column])

    return df


# Usage example
Path = input("Enter the name of the file to Automate: ")
cleaned_data = clean_data(Path)

cleaned_data.to_csv('CleanedFile.csv', index=False)
print("Data cleaning complete. Cleaned data saved to 'CleanedFile.csv'")
