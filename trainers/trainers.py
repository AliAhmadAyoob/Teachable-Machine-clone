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

# trainers/trainers.py (Modified)

from io import BytesIO
from PIL import Image
import numpy as np
import cv2 # If you use it for resizing/preproc

IMAGE_SIZE = (128, 128)

def preprocess_data(data_by_class, classes):
    """
    Loads images from file objects, preprocesses them, and prepares 
    them for training.
    """
    X = []
    y = []
    
    # Create a mapping from class name to numerical label (0, 1, 2, ...)
    class_to_label = {name: i for i, name in enumerate(classes)}
    
    for class_name, file_list in data_by_class.items():
        label = class_to_label[class_name]
        
        for file in file_list:
            try:
                # --- FIX IS HERE ---
                # Reset the file pointer to the beginning of the file.
                # This is essential because Streamlit file objects may have 
                # been read already (e.g., for showing thumbnails).
                file.seek(0) 
                
                # Use BytesIO to read the file content and open with PIL
                img = Image.open(BytesIO(file.read())).convert("RGB")
                
                # Resize the image
                img_resized = img.resize(IMAGE_SIZE)
                
                # Convert to numpy array and normalize
                img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
                
                X.append(img_array)
                y.append(label)
                
            except Exception as e:
                # Optional: Add error logging to see which file failed
                print(f"Skipping file due to error: {e}")
                continue

    return np.array(X), np.array(y)

# ... rest of trainers.py ...

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



