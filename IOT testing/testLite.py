import tensorflow as tf

model_path = 'model2.tflite'
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading TFLite model: {e}")
    exit(1)

