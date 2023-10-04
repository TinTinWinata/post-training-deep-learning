import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt

# Load data and arrange into Pandas dataframe
# Skip the first row by setting skiprows=1 (seems to be a repeated header)
df = pd.read_csv("boston.csv", header=None, skiprows=1)

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'PRICE']

df.columns = feature_names
print(df.head())

# df = df.rename(columns={'MEDV': 'PRICE'})
print(df.describe().T)

# Exploratory Data Analysis: Price, Age, and Tax Distributions
fig, ax = plt.subplots(1, 3, figsize=(95, 5))
sns.histplot(df['PRICE'], ax=ax[0])
sns.histplot(df['AGE'], ax=ax[1])
sns.histplot(df['TAX'], ax=ax[2])
plt.savefig("Exploratory Data Analysis of Price, Age, and Tax Distributions.png")
plt.show()

# Understand the data further
# Exploratory Data Analysis: Price Relationships with Age, Tax, and Black Proportion
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
sns.lineplot(x=df['AGE'], y=df['PRICE'], ax=ax[0])
sns.lineplot(x=df['TAX'], y=df['PRICE'], ax=ax[1])
sns.lineplot(x=df['B'], y=df['PRICE'], ax=ax[2])
plt.savefig("Exploratory Data Analysis of Price Relationships with Age, Tax, and Black Proportion.png")
plt.show()

# Any strong correlation between attributes?
corr = df.corr()
corr_unstacked = corr.unstack()
corr_unstacked_sorted = corr_unstacked.sort_values(kind="quicksort", ascending=False)
corr_df=pd.Series.to_frame(corr_unstacked_sorted, name='correlation')  # Convert Series to DataFrame and name correlation column accordingly. 
high_corr_features =  corr_df[corr_df.correlation != 1]  # Remove all 1s that correspond to self correlation
print(high_corr_features.head(30))

print(df.isnull().sum())
# df = df.dropna()

# Split into features and target (Price)
X = df.drop('PRICE', axis = 1)
y = df['PRICE']

# Split the data into training and temporary sets (80:20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Split the temporary set equally into validation and test sets (50:50)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=20)

total_samples = len(X)

train_proportion = len(X_train) / total_samples * 100
val_proportion = len(X_val) / total_samples * 100
test_proportion = len(X_test) / total_samples * 100

print(f"Training set proportion: {train_proportion:.2f}%")
print(f"Validation set proportion: {val_proportion:.2f}%")
print(f"Test set proportion: {test_proportion:.2f}%")

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform both the training and validation sets
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(128, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear')) # Output layer

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
print(model.summary())

# Define the number of epochs and batch size according to your needs
epochs = 100
batch_size = 16

# Fit the model using the scaled training data
history = model.fit(X_train_scaled, y_train, validation_data=(X_val_scaled, y_val), epochs=epochs, batch_size=batch_size, verbose=1)

# Plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("Training and Validation Loss.png")
plt.show()

acc = history.history['mae']
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'y', label='Training MAE')
plt.plot(epochs, val_acc, 'r', label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig("Training and Validation MAE.png")
plt.show()

# Predict on test data
predictions = model.predict(X_test_scaled[:5])
print("Predicted values are: ", predictions)
print("Real values are: ", y_test[:5])

mse_neural, mae_neural = model.evaluate(X_test_scaled, y_test)
print('Mean Squared Error  : ', mse_neural)
print('Mean Absolute Error : ', mae_neural)