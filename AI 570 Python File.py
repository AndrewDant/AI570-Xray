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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

#Upload the CSV file
path = r"C:\Users\andrew.dant\Downloads\archive"
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

#begin merging the dataset 

# Filter the DataFrame for existing images
patients_with_images = patients[patients['image_index'].isin(file_names)]

print(patients_with_images.describe())

# Loads images from all subdirectories
image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=path,
    labels=None,
    image_size=(224, 224),  # all images are resized to this size. Should match model's expectations
    batch_size=None
)


# Map image filenames to row data in patients. Reindex patients to match image order (this will throw an error if there are any image files without an associated row)
patients = patients.set_index("image_index")
patients = patients.loc[[path.split("\\")[-1] for path in image_dataset.file_paths]]
dataset_length = len(patients)


# Convert feature and label data to TensorFlow datasets
features_dataset = tf.data.Dataset.from_tensor_slices(patients[feature_columns].values)
labels_dataset = tf.data.Dataset.from_tensor_slices(patients[label_columns].values)

combined_dataset = tf.data.Dataset.zip(((image_dataset, features_dataset), labels_dataset))
# randomize the order of the dataset so we are not splitting the sets based on which folder the images are in
combined_dataset = combined_dataset.shuffle(buffer_size=combined_dataset.cardinality(), seed=1)

# Split into train and test sets (may add validation in the future)
train_size = int(0.7 * dataset_length)
test_size = dataset_length - train_size

train_dataset = combined_dataset.take(train_size)
test_dataset = combined_dataset.skip(test_size)

batch_size = 32
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
from keras.applications.resnet import ResNet50, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Input, Conv2D, MaxPooling2D, Flatten, Concatenate
from keras.utils import plot_model
from keras.optimizers import Adam

# Load the ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
image_input = base_model.input

# Freeze the layers of ResNet50 so that they are not trainable
for layer in base_model.layers:
    layer.trainable = False

# TODO does it make sense to use this architecture as well as the global average pooling layer for our problem?

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)  # Flatten the output for merging with the numerical path

# Define the numerical input path
patient_input = tf.keras.Input(shape=(len(feature_columns),), name="patient_input")
y = Dense(8, activation="relu")(patient_input)

# Concatenate the outputs from the two paths
combined = Concatenate()([x, y])

# Add additional dense layers for combined processing
z = Dense(128, activation="relu")(combined)
z = Dense(64, activation="relu")(z)

# Final output layer for multi-label classification
num_labels = len(label_columns)
output = Dense(num_labels, activation="sigmoid")(z)  # Sigmoid for multi-label binary classification

# Define the model with both inputs and the single output
model = tf.keras.Model(inputs=[image_input, patient_input], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# Display model architecture
print(model.summary())

# plot_model(model, to_file='xray_model.png')
patient_input.shape

# Train the model and save the history
history = model.fit(train_dataset, epochs=5)


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_dataset)

# Print the results
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get predictions on the test set
predictions = model.predict(test_dataset)

# Convert predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_classes = test_dataset.classes

# Generate a classification report
print(classification_report(true_classes, predicted_classes, target_names=list(test_dataset.class_indices.keys())))

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


