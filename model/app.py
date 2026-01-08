import eventlet
eventlet.monkey_patch() 
import os
import io
import time
import base64
import pickle
import numpy as np
from collections import deque 
from PIL import Image

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

"""
==============================================================================
APPLICATION CONFIGURATION & GLOBAL STATE
==============================================================================
Initializes the Flask web server and the SocketIO async mode for real-time 
communication. Defines global variables to maintain the state of the machine 
learning model, the dataset, and the training buffer used to pass data 
between the training thread and the client interface.
"""
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables to hold the loaded model and dataset in memory
model = None 
X = None
y = None

# A ring buffer (deque) used to store the most recent training metrics.
# This acts as a thread-safe bridge between the Keras training callback 
# and the frontend polling mechanism, ensuring we only serve fresh data.
training_buffer = deque(maxlen=1)

def load_data():
    """
    DATA LOADING & PREPROCESSING ROUTINE
    ------------------------------------
    This function handles the ETL (Extract, Transform, Load) pipeline for the application.
    It attempts to locate the dataset (pickle files) from a primary hardcoded path
    or a fallback directory. Once loaded, it performs critical preprocessing steps:
    1. Shuffles the data to ensure balanced training batches.
    2. Slices the data (first 1000 items) for performance optimization.
    3. Normalizes pixel values (0-255 -> 0.0-1.0).
    4. Reshapes the arrays to add channel dimensions required by Keras (N, H, W, C).
    
    If data loading fails, it generates random dummy data to allow the application
    to run in a test mode without crashing.
    """
    global X, y
    
    path_primary = "/home/nigga/engine/"
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

        X, y = shuffle(X, y, random_state=42)

        X = X[:1000] 
        y = y[:1000]

        X = X / 255.0
        y = np.array(y)

        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)

        print(f"✅ Data Loaded Successfully! Shape: {X.shape}")

    except Exception as e:
        print(f"⚠️ DATA ERROR: {e}")
        print("Using random dummy data for testing purposes.")
        X = np.random.rand(100, 50, 50, 1)
        y = np.random.randint(0, 2, 100)

# Execute data loading immediately upon server startup
load_data()

# Limit for the amount of array data sent to the frontend for visualization
VISUAL_LIMIT = 16 

def slice_data(data_array):
    """
    DATA TRUNCATION HELPER
    ----------------------
    Prepares large numpy arrays for transmission to the web frontend.
    Since sending full image matrices or layer weights is too heavy for
    real-time websocket events, this function slices the data to a 
    small "visual limit" (e.g., 16x16) suitable for UI grids.
    """
    data_array = np.array(data_array)
    if data_array.ndim == 1:
        return data_array[:VISUAL_LIMIT].tolist()
    elif data_array.ndim == 2:
        return data_array[:VISUAL_LIMIT, :VISUAL_LIMIT].tolist()
    else:
        return data_array.flatten()[:VISUAL_LIMIT].tolist()

def prepare_image(image_data, target_size):
    """
    IMAGE PREPROCESSING PIPELINE
    ----------------------------
    Converts a raw base64 image string received from the frontend into a 
    normalized numpy array formatted for the Keras model.
    Steps include:
    1. Decoding the base64 string.
    2. Converting to Grayscale to match training data.
    3. Resizing to the model's expected input dimensions.
    4. Normalizing pixel values and expanding dimensions for Batch/Channel.
    """
    if "base64," in image_data:
        image_data = image_data.split(",")[1]
    
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('L') 
    
    img = img.resize((target_size, target_size))
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = np.expand_dims(img_array, axis=-1) 
    
    return img_array

class SimulationCallback(Callback):
    """
    CUSTOM KERAS CALLBACK FOR REAL-TIME METRICS
    -------------------------------------------
    This class hooks into the model training lifecycle. At the end of every
    training batch, it captures the current loss and accuracy and appends 
    them to the global history list.
    
    Crucially, this allows us to extract training statistics continuously 
    without blocking the main training thread or the socket connection.
    """
    def on_train_batch_end(self, batch, logs=None):
        global training_history
        
        training_history.append({
            'batch': batch,
            'loss': float(logs.get('loss')),
            'accuracy': float(logs.get('accuracy'))
        })
        
    def on_epoch_end(self, epoch, logs=None):
        socketio.emit('training_progress', {'epoch': epoch + 1})

"""
==============================================================================
WEB ROUTES AND SOCKET EVENTS
==============================================================================
"""

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('pull_update')
def handle_pull_request():
    """
    FRONTEND POLLING HANDLER
    ------------------------
    This function responds to periodic requests from the frontend for update data.
    Instead of streaming every single training step (which is too fast), 
    it calculates the average loss and accuracy from the `training_buffer` 
    accumulated since the last poll. This ensures a smooth visual graph update 
    on the client side.
    """
    global training_buffer
    
    if len(training_buffer) == 0:
        return 

    count = len(training_buffer)
    avg_loss = sum(item['loss'] for item in training_buffer) / count
    avg_acc = sum(item['accuracy'] for item in training_buffer) / count
    last_batch = training_buffer[-1]['batch']

    emit('training_step', {
        'batch': last_batch,
        'loss': avg_loss,
        'accuracy': avg_acc
    })
    
    training_buffer = []

@socketio.on('start_training')
def handle_training(config_data):
    """
    MODEL TRAINING ORCHESTRATOR
    ---------------------------
    The core logic for initializing and training the neural network.
    1. Resets training history.
    2. Parses user-defined configuration (epochs, learning rate, hidden layers).
    3. Dynamically builds a Keras Sequential model based on the config.
    4. Compiles the model with the Adam optimizer.
    5. Executes `model.fit()` using the custom SimulationCallback to record progress.
    6. Emits the final history to the frontend upon completion.
    """
    global model, training_history, X, y
    
    print(f"🚀 Starting High-Speed Training. Config: {config_data}")
    
    training_history = []
    
    if X is None or y is None:
        load_data()
        if X is None:
            emit('training_error', {'error': 'Data failed to load.'})
            return

    try:
        epochs = int(config_data.get('epochs', 10))
        learning_rate = float(config_data.get('learning_rate', 0.01))
        hidden_layers = config_data.get('hidden_layers', []) 
        if not hidden_layers: 
            hidden_layers = [32] 
    except Exception as e:
        print(f"Config Error: {e}")
        emit('training_error', {'error': 'Invalid Configuration format'})
        return

    try:
        model = Sequential()
        
        model.add(Flatten(input_shape=X.shape[1:])) 
        
        for neuron_count in hidden_layers:
            model.add(Dense(int(neuron_count)))
            model.add(Activation('relu')) 
            
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        print("✅ Model compiled successfully.")

        model.fit(
            X, y,
            batch_size=32,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[SimulationCallback()], 
            verbose=1
        )
        
        print("✅ Training done. Sending history to frontend...")
        emit('training_complete_with_replay', {
            'status': 'done',
            'history': training_history
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Training Error: {e}")
        emit('training_error', {'error': str(e)})

@socketio.on('predict_sample')
def handle_prediction(data):
    """
    INFERENCE & VISUALIZATION ENGINE
    --------------------------------
    Handles requests to classify an image.
    1. Validates that a model has been trained.
    2. Preprocesses the input image.
    3. Performs binary classification (Cat vs Dog).
    4. INTROSPECTION: Runs the image through a secondary 'activation model' 
       to extract the internal outputs of every layer in the network. 
    5. Sends both the final prediction and the internal layer states to the 
       frontend for visualization.
    """
    global model
    
    if model is None:
        emit('prediction_result', {'error': 'ERROR: Model not trained yet! Please train first.'})
        return

    try:
        if hasattr(model, 'input_shape'):
            target_size = model.input_shape[1]
        elif hasattr(model.layers[0], 'input_shape'):
             target_size = model.layers[0].input_shape[1]
        else:
            target_size = 50 
            
        print(f"🔮 Predicting... Resizing to {target_size}x{target_size}")
        
        img_data = data.get('image')
        processed_img = prepare_image(img_data, target_size)
        
        prediction_score = model.predict(processed_img, verbose=0)
        
        if isinstance(prediction_score, list):
            prediction_score = prediction_score[0]
        raw_prob = float(prediction_score[0][0]) if prediction_score.ndim > 1 else float(prediction_score[0])
        
        if raw_prob > 0.5:
            label = "DOG"
            confidence = raw_prob * 100
        else:
            label = "CAT"
            confidence = (1 - raw_prob) * 100
            
        result_text = f"{label} ({confidence:.1f}%)"

        network_state = []
        try:
            layer_outputs = [layer.output for layer in model.layers]
            activation_model = Model(inputs=model.inputs, outputs=layer_outputs)
            
            all_activations = activation_model.predict(processed_img, verbose=0)
            
            if not isinstance(all_activations, list):
                all_activations = [all_activations]

            network_state.append({
                'layer_name': 'Input',
                'activations': slice_data(processed_img.flatten())
            })

            for i, val in enumerate(all_activations):
                layer_name = model.layers[i].name
                network_state.append({
                    'layer_name': layer_name,
                    'activations': slice_data(val[0]) 
                })
                
        except Exception as e_visu:
            print(f"⚠️ Visualization Error (non-fatal): {e_visu}")

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