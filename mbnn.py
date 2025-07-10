### Andrew Dant & Alicia Hernandez
### AI 570
### Professor Wang
### Chest X-Ray Project

#Libraries and stuff
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications.resnet import ResNet50, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Flatten, Concatenate
from keras.utils import plot_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Input, Concatenate, Flatten, BatchNormalization, Dropout
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import glob
import matplotlib.pyplot as plt
import os

#Upload the CSV file
path = r"D:\Penn State\AI 570\Data\Project"
os.chdir(path)
patients = pd.read_csv("Data_Entry_2017.csv")

#Data Exploration
print(patients.head())
print(patients.columns)
print(patients.describe())
print(patients.shape)  #(112120, 12)
print(patients.dtypes)     #object, int64, float64

print(patients['Patient Gender'].unique())
# encode patient gender as zero or one
patients['Patient Gender'] = patients['Patient Gender'].map({'M': 0, 'F': 1})
print(patients['View Position'].unique())
# encode patient gender as zero or one
patients['View Position'] = patients['View Position'].map({'PA': 0, 'AP': 1})
# empty/fake column caused by a trailing comma in the CSV file
columns_to_drop = ['Unnamed: 11']
all_patient_data_columns = patients
patients = patients.drop(columns=columns_to_drop)

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

feature_columns = ['patient_age', 'patient_gender', 'view_position']
print(patients[feature_columns].head())
# determine the set of unique interests
finding_set = set()
for finding_list in patients['finding_labels'].tolist():
    for finding in finding_list.split('|'):
        finding_set.add(finding)
        
print(f'Unique diagnoses in "finding_labels": {sorted(finding_set)}')

label_columns = [finding for finding in finding_set]

# build the columns and rows of the dummy variables
dummy_finding_variables = []
for finding_list in patients['finding_labels'].tolist():
    dummy_finding_variables.append([1 if finding in finding_list.split('|') else 0 for finding in finding_set])

dummy_finding_variables = pd.DataFrame(dummy_finding_variables, columns=label_columns)

# replace the old finding labels column with the dummy variables
patients.drop('finding_labels', axis=1, inplace=True)
patients = patients.join(dummy_finding_variables)
patients.dtypes

print(patients.iloc[:, -15:])



#Remove NA's
patients.isnull()   #Find missing values in data set
patients = patients.dropna()    #Drop Null values and update df

patients.describe()


#Find duplicate values
patients.duplicated().sum()     #No duplicates


patients[feature_columns]

#Correlation
# columns with a numeric datatype, other than the new dummy variables for encoding classes
numeric_features_patients = patients[feature_columns].select_dtypes(include=[np.number])
if not numeric_features_patients.empty:
    print(numeric_features_patients.corr())
else:
    print("No numeric columns to compute correlation.")     #No numeric columns to compute correlation.


#Scatterplot matrix
if not numeric_features_patients.empty and not numeric_features_patients.isnull().all(axis=0).any():
    pd.plotting.scatter_matrix(numeric_features_patients, figsize=(20, 20))
    plt.show()
else:
    print("No valid numeric data available for scatter matrix.")    #No valid numeric data available for scatter matrix.

patients['patient_age']
#Scale the data
scaler = MinMaxScaler()
patients[numeric_features_patients.columns] = scaler.fit_transform(patients[numeric_features_patients.columns])
patients.head()
file_names = []
for root, dirs, files in os.walk(path):
    file_names.extend([filename for filename in files if '.png' in filename])
    
print(len(file_names))  # the number of image files should match the number of rows in patients

# Filter the DataFrame for existing images
patients_with_images = patients[patients['image_index'].isin(file_names)]

print(patients_with_images.describe())

# Define preprocess_image
def preprocess_image(image):
    image = tf.cast(image, tf.float32)  # Convert to float32
    image = preprocess_input(image)    # ResNet50-specific preprocessing
    return image

# Get all image file paths
image_file_paths = glob.glob(r'D:\Penn State\AI 570\Data\Project\archive\**\*.png', recursive=True)

matches = [
    os.path.basename(path) in patients_with_images['image_index'].values
    for path in image_file_paths
]
print(f"First 5 matches: {matches[:5]}")
print(f"Number of matches: {sum(matches)}")

# Extract file names from file paths
patients_with_images['image_index'] = patients_with_images['image_index'].astype(str).str.strip()
image_file_names = [os.path.basename(path) for path in image_file_paths]

print(f"Number of files found: {len(image_file_paths)}")
print(image_file_paths[:5]) 

image_file_paths = glob.glob(r'D:\Penn State\AI 570\Data\Project\archive\**\*.png', recursive=True)

image_paths_filtered = [
    path for path in image_file_paths if os.path.basename(path) in patients_with_images['image_index'].values
]
print(f"Number of filtered image paths (final): {len(image_paths_filtered)}")   #Number of matches: 112120

# Function to load images
def load_image(file_path):
    img = load_img(file_path, target_size=(224, 224))  # Resize images to 224x224
    img_array = img_to_array(img)
    return preprocess_input(img_array)

def combined_data_generator(image_paths, numerical_data, labels, batch_size):
    while True:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_numerical = numerical_data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            batch_images = np.array([load_image(path) for path in batch_paths])
            yield [batch_images, batch_numerical], batch_labels

# Create a dataset of image arrays
image_paths_filtered = [path for path in image_file_paths if os.path.basename(path) in patients_with_images['image_index'].values]


print(f"Number of filtered image paths: {len(image_paths_filtered)}")
print(patients_with_images['image_index'].head())  
print(image_file_names[:5])

# Convert features and labels to NumPy arrays
features_data = patients_with_images[feature_columns].values
labels_data = patients_with_images[label_columns].values

# Function to preprocess images using TensorFlow
def preprocess_image_tf(file_path):
    img = tf.io.read_file(file_path)  # Read image file
    img = tf.image.decode_png(img, channels=3)  # Decode PNG image
    img = tf.image.resize(img, [224, 224])  # Resize to 224x224
    img = preprocess_input(img)  # ResNet50 preprocessing
    return img

# TensorFlow Dataset for images
image_dataset = tf.data.Dataset.from_tensor_slices(image_paths_filtered)
image_dataset = image_dataset.map(preprocess_image_tf, num_parallel_calls=tf.data.AUTOTUNE)

# Dataset for numerical features
numerical_dataset = tf.data.Dataset.from_tensor_slices(features_data)

# Dataset for labels
label_dataset = tf.data.Dataset.from_tensor_slices(labels_data)

# Combine all datasets into a single dataset
combined_dataset = tf.data.Dataset.zip(((image_dataset, numerical_dataset), label_dataset))

batch_size = 32

# Limit the dataset to 25,000 random observations
sample_size = 25000
patients_with_images_sampled = patients_with_images.sample(n=sample_size, random_state=42)

# Filter the image paths accordingly
image_paths_filtered_sampled = [
    path for path in image_paths_filtered if os.path.basename(path) in patients_with_images_sampled['image_index'].values
]

# Convert features and labels to NumPy arrays
features_data_sampled = patients_with_images_sampled[feature_columns].values
labels_data_sampled = patients_with_images_sampled[label_columns].values

# TensorFlow Dataset for images
image_dataset_sampled = tf.data.Dataset.from_tensor_slices(image_paths_filtered_sampled)
image_dataset_sampled = image_dataset_sampled.map(preprocess_image_tf, num_parallel_calls=tf.data.AUTOTUNE)

# Dataset for numerical features
numerical_dataset_sampled = tf.data.Dataset.from_tensor_slices(features_data_sampled)

# Dataset for labels
label_dataset_sampled = tf.data.Dataset.from_tensor_slices(labels_data_sampled)

# Combine sampled datasets
combined_dataset_sampled = tf.data.Dataset.zip(((image_dataset_sampled, numerical_dataset_sampled), label_dataset_sampled))

# Shuffle and split the dataset into training and validation sets
train_size_sampled = int(0.7 * sample_size)
train_dataset_sampled = combined_dataset_sampled.take(train_size_sampled)
val_dataset_sampled = combined_dataset_sampled.skip(train_size_sampled)

# Batch the datasets
train_dataset = train_dataset_sampled.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset_sampled.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
#Multi-Branch Neural Network Code ============================================================
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

# Define the input shapes for image and numerical data
image_input_shape = (224, 224, 3)
numerical_input_shape = (len(feature_columns),)

# Image Branch
image_input = Input(shape=image_input_shape, name="image_input")
base_model = Xception(weights="imagenet", include_top=False, input_tensor=image_input)
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
image_branch_output = BatchNormalization()(x)

# Numerical Branch
numerical_input = Input(shape=numerical_input_shape, name="numerical_input")
y = Dense(128, activation="relu")(numerical_input)
y = Dropout(0.3)(y)
numerical_branch_output = BatchNormalization()(y)

# Concatenate Branches
merged = Concatenate()([image_branch_output, numerical_branch_output])
z = Dense(128, activation="relu")(merged)
z = Dropout(0.3)(z)
z = Dense(len(label_columns), activation="sigmoid")(z)  # Multi-label classification output

# Create the model
model = Model(inputs=[image_input, numerical_input], outputs=z)

# Ensure the model tracks accuracy during training
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",  # Suitable for multi-label classification
    metrics=["accuracy"]  # Add precision metric if using Keras' built-in metrics
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5,
    callbacks=[early_stopping]
)

# Extract metrics from history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot Loss
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 3, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Calculate precision for validation set
def calculate_precision(model, dataset):
    true_labels = []
    predictions = []
    for (image_batch, numerical_batch), label_batch in dataset:
        preds = model.predict([image_batch, numerical_batch]) > 0.5  # Binary threshold
        predictions.extend(preds)
        true_labels.extend(label_batch.numpy())
    precision = precision_score(
        np.vstack(true_labels), np.vstack(predictions), average='micro'
    )
    return precision

precision_val = calculate_precision(model, val_dataset)

# Precision plot (Static value for now)
plt.subplot(1, 3, 3)
plt.bar(['Validation Precision'], [precision_val])
plt.ylim(0, 1)
plt.title('Precision on Validation Set')
plt.ylabel('Precision')

# Show all plots
plt.tight_layout()
plt.show()