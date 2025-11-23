import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
IMAGE_SIZE = (128, 128)

def train_cnn(X, y, input_shape, num_classes, epochs=5):
    """Trains a simple Convolutional Neural Network."""
    
    # 1. Define the model architecture
    model = Sequential([
        # Example architecture: Small and fast for hackathon
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax') # Final layer for classification
    ])

    # 2. Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    # 3. Train the model
    history = model.fit(X, y, epochs=epochs, verbose=0) 
    return model, history

def preprocess_data(data_dict, classes):
    """
    Reads file uploader objects, resizes images, and converts them to
    normalized NumPy arrays (X) with integer labels (y).
    """
    X = []
    y = []
    
    for i, class_name in enumerate(classes):
        for file in data_dict[class_name]:
            # Reset file pointer to the beginning
            file.seek(0)
            # Use BytesIO and file.read() to handle Streamlit uploaded file objects
            img = Image.open(BytesIO(file.read())).convert("RGB") 
            img = img.resize(IMAGE_SIZE)
            
            # Convert to array and normalize (0 to 1)
            img_array = np.asarray(img, dtype=np.float32) / 255.0 
            
            X.append(img_array)
            y.append(i) # Use the class index as the label
            
    X = np.array(X)
    y = np.array(y)
    return X, y

def split_data(X, y, test_size=0.2):
    """Splits the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def flatten_data(X):
    """Flattens image data for classical ML models."""
    num_samples = X.shape[0]
    return X.reshape(num_samples, -1)  # Flatten each image to a 1D array

def train_logistic_regression(X, y):
    """
    Trains a Logistic Regression model (classical classifier).
    Requires data to be flattened before training.
    """
    # Ensure X is flattened before fitting (2D array)
    X_flat = flatten_data(X)

    # Initialize and fit the model
    model = LogisticRegression(max_iter=100, solver='lbfgs', multi_class='auto', random_state=42)
    # Fit the model to the flattened image features
    model.fit(X_flat, y)
    return model

def train_random_forest(X, y):
    """
    Trains a Random Forest Classifier.
    Requires data to be flattened before training.
    """
    # Ensure X is flattened before fitting (2D array)
    X_flat = flatten_data(X)
    # Initialize and fit the model
    # n_estimators=100 is a reasonable default for a quick demo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_flat, y)
    return model
