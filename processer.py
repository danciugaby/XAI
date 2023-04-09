import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

def process(file_path):

    # Load the data
    data = pd.read_csv(file_path)

    # Clean the data
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data = data[data['Value'] < 1000]

    # Scale the data
    scaler = MinMaxScaler()
    data[['Value']] = scaler.fit_transform(data[['Value']])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])

    onehot_encoder = OneHotEncoder(sparse=False)
    gender_encoded = onehot_encoder.fit_transform(data[['Gender']])
    gender_encoded_df = pd.DataFrame(gender_encoded, columns=['Male', 'Female'])

    # Merge encoded data with original data and drop the original categorical column
    data = pd.concat([data, gender_encoded_df], axis=1)
    data.drop(columns=['Gender'], inplace=True)

    # Split the data
    X = data.drop(columns=['Value'])
    y = data['Value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

