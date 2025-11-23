import streamlit as st
import numpy as np
from PIL import Image
# Removed: import cv2
# Removed: from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
# Removed: import av
import uuid

# Local Imports
from trainers.trainers import train_cnn, train_logistic_regression, train_random_forest, preprocess_data, split_data
from utils import display_model_metrics

# --- PAGE CONFIG ---
st.set_page_config(page_title="Teachable Machine Clone", layout="wide", page_icon="🤖")

# --- CUSTOM CSS ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background-color: #f0f2f5;
            color: #333;
        }
        
        h1, h2, h3 {
            color: #1a73e8;
            font-weight: 700;
        }

        /* Card Styling */
        .css-1r6slb0, .stVerticalBlock {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 24px;
            font-weight: 600;
            padding: 0.5rem 1.5rem;
            transition: all 0.2s;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Headers */
        .section-header {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- STATE INITIALIZATION ---
# Check if we need to migrate or initialize
if 'classes' not in st.session_state or (st.session_state['classes'] and isinstance(st.session_state['classes'][0], str)):
    # Initialize with dict structure
    st.session_state['classes'] = [
        {'id': str(uuid.uuid4()), 'name': ''},
        {'id': str(uuid.uuid4()), 'name': ''}
    ]
    st.session_state['data'] = {} # Keyed by ID
    st.session_state['model'] = None

if 'model' not in st.session_state: st.session_state['model'] = None
if 'model_type' not in st.session_state: st.session_state['model_type'] = None

# --- HEADER ---
st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🤖 Teachable Machine Clone</h1>", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
tab1, tab2, tab3 = st.tabs(["1. Gather Data", "2. Train Model", "3. Preview"])

# ==========================================
# TAB 1: DATA COLLECTION
# ==========================================
with tab1:
    st.markdown("<div class='section-header'>Gather Data</div>", unsafe_allow_html=True)
    
    # Class Management
    for i, cls in enumerate(st.session_state['classes']):
        cls_id = cls['id']
        
        with st.container():
            # Header (show name or placeholder)
            col_header, col_remove = st.columns([4, 1])
            with col_header:
                display_name = cls['name'] if cls['name'] else f"Class {i+1}"
                st.markdown(f"### {display_name}")
            with col_remove:
                if st.button("🗑️", key=f"del_{cls_id}", help="Remove Class"):
                    st.session_state['classes'].pop(i)
                    if cls_id in st.session_state['data']:
                        del st.session_state['data'][cls_id]
                    st.rerun()
            
            # Class Name Input (Empty by default as requested)
            new_name = st.text_input("Class Name", value=cls['name'], key=f"name_{cls_id}", placeholder="Enter Class Name (e.g. Person, Dog)")
            cls['name'] = new_name # Update in place
            
            # Data Input Method (Only Upload remains)
            # The tabs UI is removed, making the uploader the direct content.
            st.markdown("### Upload Image Samples")

            uploaded_files = st.file_uploader(f"Upload for {display_name}", accept_multiple_files=True, key=f"up_{cls_id}")
            if uploaded_files:
                if cls_id not in st.session_state['data']:
                    st.session_state['data'][cls_id] = []
                # Extend list
                current_files = set(f.name for f in st.session_state['data'][cls_id])
                for uf in uploaded_files:
                    if uf.name not in current_files:
                        st.session_state['data'][cls_id].append(uf)

            # Removed: with tab_cam: block and st.camera_input
            
            # Show Count & Samples
            if cls_id in st.session_state['data'] and st.session_state['data'][cls_id]:
                count = len(st.session_state['data'][cls_id])
                st.caption(f"{count} Image Samples")
                
                # Show thumbnails
                thumbs = st.session_state['data'][cls_id][-4:] # Last 4
                cols = st.columns(4)
                for idx, file in enumerate(thumbs):
                    with cols[idx]:
                        st.image(file, use_container_width=True)
            else:
                st.info("No samples yet.")
            
            st.markdown("---")

    if st.button("➕ Add Class"):
        st.session_state['classes'].append({'id': str(uuid.uuid4()), 'name': ''})
        st.rerun()

# ==========================================
# TAB 2: TRAINING
# ==========================================
with tab2:
    st.markdown("<div class='section-header'>Train Model</div>", unsafe_allow_html=True)
    
    col_train_left, col_train_right = st.columns([1, 2])
    
    with col_train_left:
        model_option = st.selectbox("Model Type", ["CNN (Keras)", "Logistic Regression", "Random Forest"])
        
        # Training Config
        epochs = 10
        if model_option == "CNN (Keras)":
            epochs = st.slider("Epochs", 5, 50, 10)
        
        # Train Button
        if st.button("Train Model", type="primary", use_container_width=True):
            # Validation
            valid_data = True
            class_names = [c['name'] for c in st.session_state['classes']]
            
            # Check for empty names
            if any(not n.strip() for n in class_names):
                st.error("Please name all your classes before training.")
                valid_data = False
            
            # Check for duplicates
            elif len(set(class_names)) != len(class_names):
                st.error("Class names must be unique.")
                valid_data = False
            
            # Check data count
            else:
                for cls in st.session_state['classes']:
                    cid = cls['id']
                    cname = cls['name']
                    if cid not in st.session_state['data'] or len(st.session_state['data'][cid]) < 2:
                        st.error(f"Not enough data for '{cname}'. Need at least 2 images.")
                        valid_data = False
            
            if valid_data:
                with st.spinner("Training..."):
                    # Prepare data for trainer (Map ID -> Name)
                    training_data = {}
                    training_classes = []
                    for cls in st.session_state['classes']:
                        name = cls['name']
                        cid = cls['id']
                        training_classes.append(name)
                        training_data[name] = st.session_state['data'].get(cid, [])
                    
                    # Preprocess
                    X, y = preprocess_data(training_data, training_classes)
                    
                    if len(np.unique(y)) < 2:
                         st.error("Need at least 2 different classes with data.")
                    else:
                        # Split
                        X_train, X_test, y_train, y_test = split_data(X, y)
                        
                        # Train
                        if model_option == "CNN (Keras)":
                            model, history = train_cnn(X_train, y_train, X_train.shape[1:], len(training_classes), epochs=epochs)
                            st.session_state['model'] = model
                            st.session_state['model_type'] = 'cnn'
                            
                            # Plot History
                            with col_train_right:
                                st.markdown("### Training History")
                                st.line_chart(history.history['accuracy'])
                                
                                # Show Model Architecture
                                st.markdown("### Model Architecture")
                                stringlist = []
                                model.summary(print_fn=lambda x: stringlist.append(x))
                                short_model_summary = "\n".join(stringlist)
                                st.code(short_model_summary)
                                
                        elif model_option == "Logistic Regression":
                            model = train_logistic_regression(X_train, y_train)
                            st.session_state['model'] = model
                            st.session_state['model_type'] = 'sklearn'
                        elif model_option == "Random Forest":
                            model = train_random_forest(X_train, y_train)
                            st.session_state['model'] = model
                            st.session_state['model_type'] = 'sklearn'
                        
                        st.session_state['trained_classes'] = training_classes # Store for inference
                        st.success("Training Complete!")
                        
                        # Metrics
                        with col_train_right:
                            display_model_metrics(model, model_option, X_test, y_test, training_classes)

# ==========================================
# TAB 3: PREVIEW
# ==========================================
with tab3:
    st.markdown("<div class='section-header'>Preview</div>", unsafe_allow_html=True)
    
    if st.session_state['model'] is None:
        st.info("Train a model to see predictions here.")
    else:
        st.markdown("### Test Model")
        
        # Use the classes that were used during training
        inference_classes = st.session_state.get('trained_classes', [])
        
        # Only one tab remains: Upload Image
        # preview_tab_cam, preview_tab_upload = st.tabs(["📷 Webcam", "📂 Upload Image"]) # Removed
        preview_tab_upload = st.tabs(["📂 Upload Image"])[0] # Keep only the upload tab
        
        # Removed: with preview_tab_cam: block including Predictor class and webrtc_streamer call

        with preview_tab_upload:
            test_file = st.file_uploader("Upload an image to test", type=["png", "jpg", "jpeg"])
            if test_file:
                image = Image.open(test_file).convert("RGB")
                st.image(image, caption="Uploaded Image", width=300)
                
                # Preprocess
                img_array = np.asarray(image.resize((128, 128)), dtype=np.float32) / 255.0
                X_test = np.expand_dims(img_array, axis=0)
                
                # Predict
                if st.session_state['model_type'] == 'cnn':
                    probs = st.session_state['model'].predict(X_test, verbose=0)[0]
                else:
                    X_flat = X_test.reshape(1, -1)
                    if hasattr(st.session_state['model'], 'predict_proba'):
                        probs = st.session_state['model'].predict_proba(X_flat)[0]
                    else:
                        pred = st.session_state['model'].predict(X_flat)[0]
                        probs = np.zeros(len(inference_classes))
                        probs[pred] = 1.0
                
                # Display Results
                st.markdown("### Predictions")
                for i, prob in enumerate(probs):
                    if i < len(inference_classes):
                        st.progress(float(prob), text=f"{inference_classes[i]}: {prob*100:.1f}%")
