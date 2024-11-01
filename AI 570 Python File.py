### Andrew Dant & Alicia Hernandez
### AI 570
### Professor Wang
### Chest X-Ray Project

#Libraries and stuff
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import torch
from torch import nn
from torch.nn import functional as F
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
import torch.optim  as optim
import matplotlib.pyplot as plt
import os

#Upload the CSV file
path = r"D:\Penn State\AI 570\Data\Project"
os.chdir(path)
patients = pd.read_csv("Data_Entry_2017.csv")

#Data Exploration
patients.head()
patients.columns
patients.describe()
patients.shape  #(112120, 12)
patients.dtypes     #object, int64, float64

# Clean column names
patients.columns = (
    patients.columns
    .str.lower()               # Convert to lowercase
    .str.replace(' ', '_')     # Replace spaces with underscores
    .str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove special characters
    .str.strip()               # Strip whitespace
)

print("Cleaned column names:")
print(patients.columns)

patients.drop('unnamed_11', axis=1, inplace=True)

#Find unique diagnoses
unique_labels = patients['finding_labels'].unique()
print("Unique diagnoses in 'finding_labels':")
print(unique_labels)

print(patients)

#Remove NA's
patients.isnull()   #Find missing values in data set
patients = patients.dropna()    #Drop Null values and update df

patients.describe()

#Find duplicate values
patients.duplicated().sum()     #No duplicates

# Factorize categorical columns in the DataFrame
for column in patients.select_dtypes(include=['object']).columns:
    patients['finding_labels'], unique = pd.factorize(patients['finding_labels'])
print("Cleaned column names:")
print(patients.columns)

#Correlation
numeric_patients = patients.select_dtypes(include=[np.number])
if not numeric_patients.empty:
    print(numeric_patients.corr())
else:
    print("No numeric columns to compute correlation.")     #No numeric columns to compute correlation.

#Scatterplot matrix
if not numeric_patients.empty and not numeric_patients.isnull().all(axis=0).any():
    pd.plotting.scatter_matrix(numeric_patients, figsize=(20, 20))
    plt.show()
else:
    print("No valid numeric data available for scatter matrix.")    #No valid numeric data available for scatter matrix.

#Scale the data
scaler = MinMaxScaler()

numeric_patients = patients.select_dtypes(include=[np.number])
if not numeric_patients.empty:
    print("Numeric columns available for scaling:")
    print(numeric_patients.columns)
    # Scale the data
    scaler = MinMaxScaler()
    x = scaler.fit_transform(numeric_patients)  # Scale only numeric data
    # Ensure finding_labels is encoded properly
    patients['finding_labels'], unique = pd.factorize(patients['finding_labels'])
    y = patients['finding_labels']  # Use encoded labels
else:
    print("No numeric columns available for scaling.")

#begin merging the dataset 

# Filter the DataFrame for existing images
patients_with_images = patients[patients['image_file_name'].isin(os.listdir(r'D:\Penn State\AI 570\Data\Project\archive'))]

# Split into training and testing sets (70:30)
train_df, test_df = train_test_split(patients_with_images, test_size=0.3, random_state=42)

# Data augmentation for training
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=r'D:\Penn State\AI 570\Data\Project\archive',
    x_col='image_file_name',
    y_col='finding_labels',
    target_size=(150, 150),
    batch_size=100,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=r'D:\Penn State\AI 570\Data\Project\archive',
    x_col='image_file_name',
    y_col='finding_labels',
    target_size=(150, 150),
    batch_size=100,
    class_mode='categorical',
    shuffle=False  # Keep the order for testing
)

#Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=test_generator, epochs=20)

# Get features from the second-to-last layer
feature_extractor = Sequential(model.layers[:-1])  # Exclude the last layer
features = feature_extractor.predict(train_generator)

# Perform K-means clustering
n_clusters = 10  # Adjust based on your needs
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(features)

# Add clustering results to your training DataFrame
train_df['cluster'] = kmeans.labels_

print("Clustering results added to training DataFrame:")
print(train_df[['image_file_name', 'finding_labels', 'cluster']].head())


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_generator)

# Train the model and save the history
history = model.fit(train_generator, validation_data=test_generator, epochs=20)

# Print the results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get predictions on the test set
predictions = model.predict(test_generator)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = test_generator.classes

# Generate a classification report
print(classification_report(true_classes, predicted_classes, target_names=list(test_generator.class_indices.keys())))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


