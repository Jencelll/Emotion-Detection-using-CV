
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

try:
    model = load_model('emotion_detector_model.h5')
    print("Input Shape:", model.input_shape)
    print("Output Shape:", model.output_shape)
except Exception as e:
    print("Error loading model:", e)
