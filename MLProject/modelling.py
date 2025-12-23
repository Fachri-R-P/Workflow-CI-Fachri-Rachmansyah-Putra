import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv(
    '../MLProject/preprocessing/airquality_preprocessing.csv'
)

# Pisahkan fitur & target
X = df.drop(columns=['CO(GT)', 'Datetime'], errors='ignore')
y = df['CO(GT)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Aktifkan MLflow autolog
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("RMSE:", rmse)
    print("R2:", r2)
