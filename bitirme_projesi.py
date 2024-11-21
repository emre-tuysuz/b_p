import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv("Imports_Exports.csv")
print("Veri setinin ilk satırları:")
print(dataset.head())

# Eksik değerleri kontrol et! Doldur!
for col in ['Country', 'Product', 'Shipping_Method', 'Port', 'Category']:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True) 
for col in ['Quantity', 'Value', 'Weight']:
    dataset[col].fillna(dataset[col].mean(), inplace=True)

# One-Hot Encoding
dataset = pd.get_dummies(dataset, columns=['Country', 'Product', 'Shipping_Method', 'Port', 'Category'], drop_first=True)

numerical_features = ['Quantity', 'Value', 'Weight']
scaler = StandardScaler()
dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])

X = dataset.drop('Import_Export', axis=1)  # Özellikler
y = dataset['Import_Export']  # Hedef sütun

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Eğitim veri seti boyutu: {X_train.shape}")
print(f"Test veri seti boyutu: {X_test.shape}")



y = y.map({'Import': 0, 'Export': 1})

# Label Encoding işlemi
label_encoder = LabelEncoder()

for col in X_train.columns:
    if X_train[col].dtype == 'object':

        X_train[col] = label_encoder.fit_transform(X_train[col])

        unseen_categories = ~X_test[col].isin(label_encoder.classes_)
        X_test.loc[unseen_categories, col] = 'unknown'  
        

        label_encoder.classes_ = np.append(label_encoder.classes_, 'unknown')
        

        X_test[col] = label_encoder.transform(X_test[col])

# array a çeviriyoruz np ile
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy * 100:.2f}%")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))



