from tensorflow.keras.models import model_from_json

# Load the model architecture from a JSON file
with open("model_a1.json", "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

# Load the weights into the model
model.load_weights("model_weights1.h5")
