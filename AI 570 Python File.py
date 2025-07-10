### Andrew Dant & Alicia Hernandez
### AI 570
### Professor Wang
### Chest X-Ray Project

#Libraries and stuff
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, multilabel_confusion_matrix
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

label_columns = ['any_finding']

# build the columns and rows of the dummy variables
dummy_finding_variables = []
for finding_list in patients['finding_labels'].tolist():
    dummy_finding_variables.append([0 if "No Finding".lower() in finding_list.lower() else 1])

dummy_finding_variables = pd.DataFrame(dummy_finding_variables, columns=label_columns)

# replace the old finding labels column with the dummy variables
patients.drop('finding_labels', axis=1, inplace=True)
patients = patients.join(dummy_finding_variables)
patients.dtypes

print(patients[label_columns].head())

patients[label_columns].sum()
# Analyze class imbalance 
total_count = len(patients)
print(f"Number of items: {total_count}")
for label in label_columns:
    true_positive_count = patients[label].sum()
    print(f"{label:<20}: {true_positive_count/total_count:.4f} % positive")
# Compute class weights to help address class imbalance

class_counts = np.sum(patients[label_columns], axis=0)

class_weights = total_count / (len(class_counts) * class_counts)

class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(class_weight_dict)

#Remove NA's
patients.isnull()   #Find missing values in data set
patients = patients.dropna()    #Drop Null values and update df

patients.describe()


#Find duplicate values
patients.duplicated().sum()     #No duplicates


patients[feature_columns]

#Correlation
# feature columns with a numeric datatype
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

patients['patient_age'].describe()
# maximum patient age does not make sense, most likely bad data since values were obtained via text mining
# clipping the maximum age at 120 years
patients['patient_age'] = patients['patient_age'].clip(upper=120)
patients['patient_age'].describe()
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

batch_size = 32

# Loads images from all subdirectories
image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=path,
    labels=None,
    image_size=(224, 224),  # all images are resized to this size. Should match model's expectations
    batch_size=None
)
# Map image filenames to row data in patients. Reindex patients to match image order (this will throw an error if there are any image files without an associated row)
reindexed_patients = patients.set_index("image_index")
image_mapped_patients = reindexed_patients.loc[[path.split("\\")[-1] for path in image_dataset.file_paths]]
dataset_length = len(image_mapped_patients)


# Convert feature and label data to TensorFlow datasets
features_dataset = tf.data.Dataset.from_tensor_slices(image_mapped_patients[feature_columns].values)
labels_dataset = tf.data.Dataset.from_tensor_slices(image_mapped_patients[label_columns].values)

combined_dataset = tf.data.Dataset.zip(((image_dataset, features_dataset), labels_dataset))
image_and_labels_dataset = tf.data.Dataset.zip((image_dataset, labels_dataset))
# randomize the order of the dataset so we are not splitting the sets based on which folder the images are in
combined_dataset = combined_dataset.shuffle(buffer_size=1000, seed=1)
image_and_labels_dataset = image_and_labels_dataset.shuffle(buffer_size=1000, seed=1)

# Split into train and test sets (may add validation in the future)
train_size = int(0.7 * dataset_length)
test_size = dataset_length - train_size
print(f'train_size: {train_size}')

combined_train_dataset = combined_dataset.take(train_size)
combined_test_dataset = combined_dataset.skip(train_size)

combined_train_dataset = combined_train_dataset.batch(batch_size)
combined_test_dataset = combined_test_dataset.batch(batch_size)

combined_train_dataset = combined_train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
combined_test_dataset = combined_test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

image_and_labels_train_dataset = image_and_labels_dataset.take(train_size)
image_and_labels_test_dataset = image_and_labels_dataset.skip(train_size)

image_and_labels_train_dataset = image_and_labels_train_dataset.batch(batch_size)
image_and_labels_test_dataset = image_and_labels_test_dataset.batch(batch_size)

image_and_labels_train_dataset = image_and_labels_train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
image_and_labels_test_dataset = image_and_labels_test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

from keras.applications.resnet import ResNet50, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Concatenate, Dropout
from keras.optimizers import Adam

def build_and_compile_ResNet50_CNN_only():
    
    # Load the ResNet50 model without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    image_input = base_model.input

    # Freeze the layers of ResNet50 so that they are not trainable
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    
    x = Flatten()(x)
    x = Dense(8)(x)
    x = Dropout(0.3)(x)
    
    # Final output layer for multi-label classification
    num_labels = len(label_columns)
    output = Dense(num_labels, activation="sigmoid")(x)  # Sigmoid for multi-label binary classification

    resnet_model = tf.keras.Model(inputs=[image_input], outputs=output)

    resnet_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    # Display model architecture
    print(resnet_model.summary())
    
    return resnet_model

from keras.applications.resnet import ResNet50, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Concatenate, Dropout
from keras.optimizers import Adam

def build_and_compile_ResNet50():
    
    # Load the ResNet50 model without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    image_input = base_model.input

    # Freeze the layers of ResNet50 so that they are not trainable
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)  # Flatten the output for merging with the numerical path

    # # Define the numerical input path
    patient_input = tf.keras.Input(shape=(len(feature_columns),), name="patient_input")
    y = Dense(8, activation="relu")(patient_input)

    # # Concatenate the outputs from the two paths
    combined = Concatenate()([x, y])

    # Fully connected layers for the merged path
    z = Dense(64, activation="relu")(combined)
    z = Dropout(0.3)(z)
    z = Dense(32, activation="relu")(z)

    # Final output layer for multi-label classification
    num_labels = len(label_columns)
    output = Dense(num_labels, activation="sigmoid")(z)  # Sigmoid for multi-label binary classification

    # Define the model with both inputs and the single output
    resnet_model = tf.keras.Model(inputs=[image_input, patient_input], outputs=output)

    # Compile the model
    resnet_model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

    # Display model architecture
    print(resnet_model.summary())
    
    return resnet_model

# from keras.utils import plot_model
# plot_model(model, to_file='xray_model.png')
def display_history_graphs(history):
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        
        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        # plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()
import os

def train_or_load_model(training_dataset, build_function, model_file_path):
    model = None
    if os.path.exists(model_file_path):
        print("Loading saved model.")
        
        model = tf.keras.models.load_model(model_file_path)
    else:
        print("No saved model found. Training new model.")
        model = build_function()
        history = model.fit(
            training_dataset,
            epochs=10
        )
        # Save the trained weights
        print("Saving new model weights.")
        model.save(model_file_path)

        display_history_graphs(history)
    
    return model
import codecs, json

def generate_or_retrieve_predictions(model, test_dataset, prediction_filepath):
    predictions = None

    if os.path.exists(prediction_filepath):
        print("Loading saved predictions.")
        with open(prediction_filepath, 'r') as predictions_file:
            stored_predictions = json.load(codecs.open(prediction_filepath, 'r', encoding='utf-8'))
            predictions = np.asarray(stored_predictions)
    else:
        print('No saved predictions found. Generating predictions on test dataset.')
        predictions = model.predict(test_dataset)
        
        print('Saving new predictions.')
        json.dump(predictions.tolist(), codecs.open(prediction_filepath, 'w', encoding='utf-8'))
        
    return predictions
from sklearn.metrics import roc_curve, roc_auc_score

# we don't need to do this multiple times assuming all test datasets are the same size
y_true = np.concatenate([y for x, y in combined_test_dataset], axis=0)

# display a number of metrics based on a model's predictions
def display_prediction_results(predictions, test_dataset):
    # Number of labels
    num_labels = y_true.shape[1]

    # Initialize figure for ROC curves
    plt.figure(figsize=(10, 8))

    # Compute ROC curve and AUC for each label
    for i in range(num_labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], predictions[:, i])
        auc = roc_auc_score(y_true[:, i], predictions[:, i])
        plt.plot(fpr, tpr, label=f'Label {i} (AUC = {auc:.2f})')

    # Finalize plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Label Classification')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()

    threshold = 0.2
    binary_predictions = (predictions > threshold).astype(int)

    # Generate a classification report
    print(classification_report(y_true, binary_predictions))
    for i, confusion_matrix in enumerate(multilabel_confusion_matrix(y_true, binary_predictions)):
        print(label_columns[i])
        print(confusion_matrix)
CNN_resnet_model = train_or_load_model(image_and_labels_train_dataset, build_and_compile_ResNet50_CNN_only, "ResNet50_CNN_NoFindings.h5")
CNN_ResNet50_predictions = generate_or_retrieve_predictions(CNN_resnet_model, image_and_labels_test_dataset, 'ResNet50_CNN_NoFindings_predictions.json')
display_prediction_results(CNN_ResNet50_predictions, image_and_labels_test_dataset)