import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Assuming your data is in a DataFrame called df
# Replace this with your actual loading mechanism
# df = pd.read_csv("path_to_file.csv")

# Convert date columns to ordinal
date_cols = ['RECL_DATE_SINISTRE_DT', 'RECL_DATE_RAPPORTEE_DT', 'COND_PERMIS_DATE_DT']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce').apply(lambda x: x.toordinal())

# Handle missing values (simple imputation for demonstration)
df.fillna(df.mean(numeric_only=True), inplace=True)  # Numeric columns: fill with mean
df.fillna("Unknown", inplace=True)  # Categorical columns: fill with "Unknown"

# Convert categorical columns to numeric using LabelEncoder
categorical_cols = df.select_dtypes(include='object').columns.tolist()
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Separate features and target variable
X = df.drop("RECL_COUT_REPARATION_NUM", axis=1)
y = df["RECL_COUT_REPARATION_NUM"]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply RFE
model = RandomForestRegressor()
selector = RFE(model, n_features_to_select=20)
selector = selector.fit(X_train, y_train)

# Get the top features
top_features = X.columns[selector.support_].tolist()
print("Top 20 Features:", top_features)
