# utils.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def flatten_data(X):
    """Flatten image arrays for classical ML models."""
    return X.reshape(X.shape[0], -1)

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def display_model_metrics(model, model_name, X_test, y_test, classes):
    """Display accuracy and confusion matrix for any model."""
    if "CNN" in model_name:  # CNN
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    else:
        y_pred = model.predict(flatten_data(X_test))

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader(f"✅ {model_name} Results")
    st.metric("Accuracy", f"{accuracy*100:.2f}%")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix - {model_name}")
    st.pyplot(fig)
