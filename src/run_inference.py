import argparse
import pandas as pd
import os
from preprocess import generateData
from keras.utils import pad_sequences
import tensorflow as tf
import pickle

from run_training import get_f1


def main(model_id=None):

    if model_id is None:
        model_id = input("Model ID: ")

    model_name = f"Model_ID{model_id}.h5"
    print("Using model ", model_name)

    validation_text = generateData(
        os.path.join("data", "raw", "test.csv"), dataset_type="test"
    )
    print("validation data read in: ", len(validation_text))

    outputs_path = "models"

    file_name = f"{model_id}_tokenizer.pickle"
    full_path = os.path.join(outputs_path, file_name)
    with open(full_path, "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Loaded tokenizer")

    file_name = f"{model_id}_model.h5"
    full_path = os.path.join(outputs_path, file_name)
    model = tf.keras.models.load_model(full_path, custom_objects={"get_f1": get_f1})
    print("Loaded model")

    validation_text = tokenizer.texts_to_sequences(validation_text)
    validation_text = pad_sequences(validation_text, padding="post", maxlen=32)

    results = model.predict(validation_text)

    # convert to class
    threshold = 0.5
    results = [1 if x >= threshold else 0 for x in results]
    print("Predictions generated")

    # convert to kaggle
    validation_data_in = pd.read_csv(os.path.join("data", "raw", "test.csv"))
    validation_data_in["target"] = results
    validation_data_out = validation_data_in[["id", "target"]]

    file_path = os.path.join("outputs", f"{model_id}_submission.csv")
    validation_data_out.to_csv(file_path, index=False)
    print("Output saved: ", file_path)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaster Tweets inference script.")
    parser.add_argument("--id", action="store", default=None, dest="model_id")
    args = parser.parse_args()
    # e.g. model_id = "20221013214858_lstm"
    main(args.model_id)
