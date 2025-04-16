import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model():
    model = load_model('keras_model.h5')
    test_data_path = './Dataset/'
    
    data_gen = ImageDataGenerator(rescale=1./255)
    test_generator = data_gen.flow_from_directory(
        test_data_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

if __name__ == "__main__":
    evaluate_model()
