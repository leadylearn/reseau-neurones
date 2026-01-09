import eventlet
eventlet.monkey_patch() 
import os
import io
import time
import base64
import pickle
import numpy as np
from collections import deque # <--- 1. Import deque
from PIL import Image

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables
model = None 
X = None
y = None
# --- NEW: RING BUFFER ---
# maxlen=1 ensures we only keep the absolute latest frame. 
# Old frames are automatically discarded if the frontend hasn't asked for them yet.
training_buffer = deque(maxlen=1)
# --- 1. DATA LOADING ---
def load_data():
    global X, y
    
    # 1. Try your specific absolute path
    path_primary = "/home/nigga/engine/"
    # 2. Fallback to the current directory where app.py is
    path_fallback = os.path.dirname(os.path.abspath(__file__))

    try:
        print("Attempting to load data...")
        
        if os.path.exists(os.path.join(path_primary, "X1.pickle")):
            print(f"Loading from {path_primary}...")
            with open(os.path.join(path_primary, "X1.pickle"), "rb") as f: X = pickle.load(f)
            with open(os.path.join(path_primary, "y1.pickle"), "rb") as f: y = pickle.load(f)
        else:
            print(f"Primary path not found. Loading from {path_fallback}...")
            with open(os.path.join(path_fallback, "X1.pickle"), "rb") as f: X = pickle.load(f)
            with open(os.path.join(path_fallback, "y1.pickle"), "rb") as f: y = pickle.load(f)

        # --- CRITICAL: SHUFFLE DATA ---
        # Ensures we don't just train on one class if the data is sorted
        X, y = shuffle(X, y, random_state=42)

        # Slice for performance (Train on first 1000 images for speed)
        X = X[:1000] 
        y = y[:1000]

        # Normalize
        X = X / 255.0
        y = np.array(y)

        # Ensure correct shape (N, H, W, Channels)
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)

        print(f"✅ Data Loaded Successfully! Shape: {X.shape}")

    except Exception as e:
        print(f"⚠️ DATA ERROR: {e}")
        print("Using random dummy data for testing purposes.")
        X = np.random.rand(100, 50, 50, 1)
        y = np.random.randint(0, 2, 100)

# Load data immediately on startup
load_data()

# --- 2. HELPERS ---
VISUAL_LIMIT = 16 

def slice_data(data_array):
    """Slices matrices that are too big for the web visualization."""
    data_array = np.array(data_array)
    if data_array.ndim == 1:
        return data_array[:VISUAL_LIMIT].tolist()
    elif data_array.ndim == 2:
        return data_array[:VISUAL_LIMIT, :VISUAL_LIMIT].tolist()
    else:
        return data_array.flatten()[:VISUAL_LIMIT].tolist()

def prepare_image(image_data, target_size):
    """Converts base64 web image to numpy array formatted for the model."""
    if "base64," in image_data:
        image_data = image_data.split(",")[1]
    
    img_bytes = base64.b64decode(image_data)
    # Convert to Grayscale ('L') to match X1.pickle data
    img = Image.open(io.BytesIO(img_bytes)).convert('L') 
    
    img = img.resize((target_size, target_size))
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # Add Batch dimension and Channel dimension: (1, 50, 50, 1)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = np.expand_dims(img_array, axis=-1) 
    
    return img_array

# --- 3. OPTIMIZED CALLBACK ---
class SimulationCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        # INSTEAD of emitting, we save to the buffer
        global training_buffer
        
        training_buffer.append({
            'batch': batch,
            'loss': float(logs.get('loss')),
            'accuracy': float(logs.get('accuracy'))
        })
        
        # A tiny sleep allows the Eventlet loop to process the 'pull_update' request
        # from the frontend even while training is hogging the CPU.
        eventlet.sleep(0.0)

    def on_epoch_end(self, epoch, logs=None):
        # Epochs are slow enough to emit directly
        socketio.emit('epoch_update', {
            'epoch': epoch + 1,
            'loss': float(logs.get('loss')),
            'accuracy': float(logs.get('accuracy')),
            'val_loss': float(logs.get('val_loss', 0)),
            'val_accuracy': float(logs.get('val_accuracy', 0))
        })
# --- 4. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')
# --- NEW: POLLING ENDPOINT ---
@socketio.on('pull_update')
def handle_pull_request():
    """Frontend asks for the latest data here"""
    if len(training_buffer) > 0:
        # Get the latest item (peak at the end of the deque)
        data = training_buffer[-1]
        emit('training_step', data)
@socketio.on('start_training')
def handle_training(config_data):
    global model, training_buffer
    
    # Clear buffer before starting
    training_buffer.clear()
    
    print(f"🚀 Starting Training. Config: {config_data}")
    
    # Get dimensions from loaded data
    # (Ensure X is loaded. If using dummy data, shape might be different)
    if X is None:
        emit('training_error', {'error': 'Data not loaded on server.'})
        return

    real_height = X.shape[1] 
    real_width = X.shape[2]
    real_channels = X.shape[3]
    
    learning_rate = float(config_data.get("learning_rate", 0.001))
    epochs = int(config_data.get('epochs', 10))
    
    # FIX: Read the simple list of numbers sent from JS
    hidden_layers = config_data.get("hidden_layers", []) 
    
    model = Sequential()
    
    # 1. Input Flatten Layer
    model.add(Flatten(input_shape=(real_height, real_width, real_channels), name="flatten"))

    # 2. Dynamic Dense Layers (The fix)
    # We loop through the list [16, 16, 16] sent from JS
    for i, units in enumerate(hidden_layers):
        model.add(Dense(int(units), name=f"dense_{i}"))
        model.add(Activation("relu"))

    # 3. Output Layer
    model.add(Dense(1, activation="sigmoid", name="output"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    try:
        model.fit(
            X, y,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[SimulationCallback()],
            verbose=1
        )
        emit('training_complete', {'status': 'done'})
    except Exception as e:
        print(f"❌ Training Error: {e}")
        emit('training_error', {'error': str(e)})
@socketio.on('predict_sample')
def handle_prediction(data):
    global model
    
    if model is None:
        emit('prediction_result', {'error': 'ERROR: Model not trained yet! Please train first.'})
        return

    try:
        # Determine expected input size from the first layer
        if hasattr(model, 'input_shape'):
            target_size = model.input_shape[1]
        elif hasattr(model.layers[0], 'input_shape'):
             target_size = model.layers[0].input_shape[1]
        else:
            target_size = 50 # Fallback
            
        print(f"🔮 Predicting... Resizing to {target_size}x{target_size}")
        
        img_data = data.get('image')
        processed_img = prepare_image(img_data, target_size)
        
        # 1. Prediction
        prediction_score = model.predict(processed_img, verbose=0)
        
        # Unpack result
        if isinstance(prediction_score, list):
            prediction_score = prediction_score[0]
        raw_prob = float(prediction_score[0][0]) if prediction_score.ndim > 1 else float(prediction_score[0])
        
        # 2. Logic: Cat (0) vs Dog (1)
        if raw_prob > 0.5:
            label = "DOG"
            confidence = raw_prob * 100
        else:
            label = "CAT"
            confidence = (1 - raw_prob) * 100
            
        result_text = f"{label} ({confidence:.1f}%)"

        # 3. Visualization (Extract Activations)
        network_state = []
        try:
            # Create a sub-model that outputs every layer's result
            layer_outputs = [layer.output for layer in model.layers]
            activation_model = Model(inputs=model.inputs, outputs=layer_outputs)
            
            all_activations = activation_model.predict(processed_img, verbose=0)
            
            if not isinstance(all_activations, list):
                all_activations = [all_activations]

            # Input Vis
            network_state.append({
                'layer_name': 'Input',
                'activations': slice_data(processed_img.flatten())
            })

            # Layer Vis
            for i, val in enumerate(all_activations):
                layer_name = model.layers[i].name
                # We only visualize layers that have meaningful output
                network_state.append({
                    'layer_name': layer_name,
                    'activations': slice_data(val[0]) 
                })
                
        except Exception as e_visu:
            print(f"⚠️ Visualization Error (non-fatal): {e_visu}")

        # 4. Emit
        emit('prediction_simulation', {
            'network_state': network_state,
            'probability': raw_prob,
            'label': label,
            'confidence': f"{confidence:.1f}%",
            'text': result_text
        })
        
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        emit('prediction_result', {'error': str(e)})

if __name__ == '__main__':
    print("🚀 Server starting on http://127.0.0.1:5000")
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)