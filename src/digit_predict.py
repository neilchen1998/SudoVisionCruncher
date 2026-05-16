import cv2
import numpy as np
import os
import tensorflow as tf

def load_model(model_path: str) -> tf.keras.Model:
    """
    Loads the tensorflow model

    Args:
        model_path: The path of the model

    Returns:
        tf.keras.Model: The loaded TensorFlow Keras model
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path)

        print("Model loaded successfully!")
        return model

    except Exception as e:
        print(f"Failed to load the model: {e}")
        raise

def load_and_preprocess_image(image_path, target_size=(28, 28), inverse=True):
    """
    Loads an image in grey scale, resizes it, optionally inverts the colour,
    normalizes it, and reshapes it for the model

    Args:
        image_path: The path of the original image
        target_size: The target size of the image (based on how the model was trained)
        inverse: Inverses the digit from black to white

    Returns:
        NDArray: The processed image
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image can be loaded properly
    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    # Resize the image based on the target size
    img_ret = cv2.resize(img, target_size)

    # Invert the colour if necessary since digits in MNIST are white on a black background
    if inverse:
        img_ret = cv2.bitwise_not(img_ret)

    # Normalize the pixel values to [0, 1]
    img_ret = img_ret.astype('float32') / 255.0

    img_ret = np.reshape(img_ret, (-1, target_size[0], target_size[1], 1))

    return img_ret

def predict_digit(model: tf.keras.models, img) -> tuple[int, float]:
    """
    Predicts the digit and output a confidence score

    Args:
        model: The Keras model

    Returns:
        tuple[int, float]: [the index of the predicted digit, the confidence score]
    """

    predictions = model.predict(img, verbose=0)

    predicted_digit = np.argmax(predictions, axis=-1)[0]
    confidence = np.max(predictions) * 100

    return predicted_digit, confidence

def run_digit_prediction_pipeline(model_path, image_path):
    """
    Runs the entire pipeline that predicts the digit and output a confidence score

    Args:
        model_path: The path of the model
        image_path: The path of the original image

    Returns:
        tuple[int, float]: [the index of the predicted digit, the confidence score]
    """

    digit_model = load_model(model_path)

    img_final = load_and_preprocess_image(image_path)

    return predict_digit(digit_model, img_final)
