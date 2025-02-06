import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

train_data_path = Path('../input/train.csv')
test_data_path = Path('../input/test.csv')

try:
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    raise

y = train_data['SalePrice']
features = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
X = train_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(train_X, train_y)

test_X = test_data[features]
test_preds = model.predict(test_X)

output = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': test_preds})
output_path = Path('submission.csv')
output.to_csv(output_path, index=False)

logging.info(f"Predictions saved to {output_path}")
