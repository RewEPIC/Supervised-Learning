
from keras.models import load_model

def load(model_path = "models/keras_model.h5", compile = True):
    model = load_model(filepath=model_path, compile=compile) # TensorFlow is required for Keras to work
    class_names = [line.strip()[line.find(' ') + 1:] for line in open("models/labels.txt", "r").readlines()]
    return model, class_names