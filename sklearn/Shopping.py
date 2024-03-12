import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv(r"C:\Users\Hari Keshav Rajesh\Desktop\Computer Projects and resources\Datasets\Shopping_Behaviour\shopping_behavior_updated.csv")

'''
List of Columns:
Age,Gender,Location,Season,Subscription Status,Shipping Type,Discount Applied,Promo Code Used,Previous Purchases,Payment Method,Frequency of Purchases

Factor to be predicted: 
Review Rating
'''

imputer = SimpleImputer(strategy='mean')
ord_encoder  = OrdinalEncoder()

df[['Gender_encoded', 'Location_encoded','Season_encoded','Subscription Status_encoded','Shipping Type_encoded','Discount Applied_encoded','Promo Code Used_encoded','Previous Purchases_encoded','Payment Method_encoded','Frequency of Purchases_encoded']] = ord_encoder.fit_transform(df[['Gender', 'Location','Season','Subscription Status','Shipping Type','Discount Applied','Promo Code Used','Previous Purchases','Payment Method','Frequency of Purchases']])

test_df = df[['Gender_encoded', 'Location_encoded','Season_encoded','Subscription Status_encoded','Shipping Type_encoded','Discount Applied_encoded','Promo Code Used_encoded','Previous Purchases_encoded','Payment Method_encoded','Frequency of Purchases_encoded']]

X = pd.DataFrame(imputer.fit_transform(test_df),columns=test_df.columns)
y = df['Review Rating']

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = RandomForestRegressor(random_state=1)

model.fit(train_X,train_y)

model_preds = model.predict(val_X)
final = model_preds.mean()
model_mae =  mean_absolute_error(val_y,model_preds)

print("The mean rating is: ",final)
print("The mean absolute error is: ",model_mae)




