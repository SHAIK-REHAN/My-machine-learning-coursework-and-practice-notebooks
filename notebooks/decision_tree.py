import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

train_df = pd.read_parquet("train.parquet")
test_df = pd.read_parquet("test.parquet")

print("\nDATA PREVIEW\n")
print(train_df.head())

print("\nDATA INFO\n")
print(train_df.info())

print("\nDATA TYPES\n")
print(train_df.dtypes)

print("\nSTATISTICAL SUMMARY\n")
print(train_df.describe())

print("\nMISSING VALUES\n")
print(train_df.isnull().sum())

for df in [train_df, test_df]:
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["week"] = df["Date"].dt.isocalendar().week.astype("int")
    df.drop(columns=["Date"], inplace=True)

train_df = train_df.dropna()
test_df = test_df.fillna(0)

target = "target"

encoder = LabelEncoder()
train_df[target] = encoder.fit_transform(train_df[target])

X = train_df.drop(columns=[target])
y = train_df[target]

X = pd.get_dummies(X)
test_df = pd.get_dummies(test_df)

X, test_df = X.align(test_df, axis=1, fill_value=0)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
test_df = test_df.apply(pd.to_numeric, errors="coerce").fillna(0)

print("\nFINAL FEATURE TYPES\n")
print(X.dtypes.value_counts())

X = X.sample(n=250000, random_state=42)
y = y.loc[X.index]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = HistGradientBoostingClassifier(
    max_depth=10,
    learning_rate=0.08,
    max_iter=300,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_val)
accuracy = accuracy_score(y_val, pred)

print("\nMODEL ACCURACY\n")
print(accuracy)

final_predictions = model.predict(test_df)

print("\nTEST PREDICTIONS SAMPLE\n")
print(final_predictions[:10])
