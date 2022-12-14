import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from keras import layers
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import keras.backend as K

import logging
import datetime
import os
import pprint
import matplotlib.pyplot as plt
import pickle

from preprocess import generateData


def getModelID(model_type):
    """Get unique ID for model based on datetime"""
    datetime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    id_value = f"{datetime_str}_{model_type}"
    return id_value


def getMetrics(Y_train, Y_train_pred, Y_test, Y_test_pred, model_id):

    metrics_list = [
        ("accuracy_score", metrics.accuracy_score),
        ("precision_score", metrics.precision_score),
        ("recall_score", metrics.recall_score),
        ("f1_score", metrics.f1_score),
        ("confusion_matrix", metrics.confusion_matrix),
    ]
    train_pred_sets = [
        ("train", Y_train, Y_train_pred),
        ("test", Y_test, Y_test_pred),
    ]

    metrics_results = {"model_id": model_id}
    for metric_name, metric in metrics_list:
        for name, Y, Y_pred in train_pred_sets:
            metrics_results[f"{metric_name}_{name}"] = metric(Y, Y_pred)

    return metrics_results


def saveMetrics(metrics_results, outputs_path="outputs"):

    model_id = metrics_results["model_id"]
    file_name = f"{model_id}_metrics.txt"
    full_path = os.path.join(outputs_path, file_name)

    with open(full_path, "w") as text_file:
        # pretty print for readability in file
        text_file.write(pprint.pformat(metrics_results))
    logging.info(f"Model metrics saved as: {file_name}")
    return None


def saveModel(model, model_id, outputs_path="models"):
    file_name = f"{model_id}_model.h5"
    full_path = os.path.join(outputs_path, file_name)
    model.save(full_path)
    logging.info(f"Model saved as: {file_name}")
    return None


def saveTokenizer(tokenizer, model_id, outputs_path="models"):
    file_name = f"{model_id}_tokenizer.pickle"
    full_path = os.path.join(outputs_path, file_name)
    logging.info(f"Model tokenizer saved as: {file_name}")
    # saving tokenizer
    with open(full_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return None


def get_f1(y_true, y_pred):
    # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1_val


# naive bayes algorithm
def naiveBayes(X_train, Y_train, X_test, Y_test):

    model_id = getModelID("naiveBayes")
    logging.info(f"Model ID: {model_id}")

    # Create naive bayes Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(X_train, Y_train)

    # predict on train dataset
    Y_train_pred = gnb.predict(X_train)

    # Predict the response for test dataset
    Y_test_pred = gnb.predict(X_test)

    metrics_results = getMetrics(Y_train, Y_train_pred, Y_test, Y_test_pred, model_id)

    pprint.pprint(metrics_results)
    logging.info(f'accuracy_score_train: {metrics_results["accuracy_score_train"]}')
    logging.info(f'accuracy_score_test: {metrics_results["accuracy_score_test"]}')
    logging.info(f'f1_score_train: {metrics_results["f1_score_train"]}')
    logging.info(f'f1_score_test: {metrics_results["f1_score_test"]}')

    saveMetrics(metrics_results, outputs_path="outputs")

    return gnb, model_id


# SVM algorithm
def svmCode(X_train, Y_train, X_test, Y_test):

    model_id = getModelID("svmCode")
    logging.info(f"Model ID: {model_id}")

    # creating the SVM model and setting its parameters
    model = svm.SVC(kernel="linear", C=100, gamma=1e-7)

    # training the model
    model.fit(X_train, Y_train)

    # predicts the labels of the training data
    predicted_labels_train = model.predict(X_train)

    # predicts the labels of the test images
    predicted_labels = model.predict(X_test)

    # generate accuracy of SVM
    accuracy = metrics.accuracy_score(Y_test, predicted_labels) * 100

    print("The level of accuracy is: " + str(accuracy) + "%")

    print("\nThe confusion matrix: ")
    # generate confusion matrix on results
    print(metrics.confusion_matrix(Y_test, predicted_labels))

    print("\nThe classification report: ")
    # generate classification table on results
    print(metrics.classification_report(Y_test, predicted_labels))

    metrics_results = getMetrics(
        Y_train,
        predicted_labels_train,
        Y_test,
        predicted_labels,
        model_id,
    )

    pprint.pprint(metrics_results)
    logging.info(f'accuracy_score_train: {metrics_results["accuracy_score_train"]}')
    logging.info(f'accuracy_score_test: {metrics_results["accuracy_score_test"]}')
    logging.info(f'f1_score_train: {metrics_results["f1_score_train"]}')
    logging.info(f'f1_score_test: {metrics_results["f1_score_test"]}')

    saveMetrics(metrics_results, outputs_path="outputs")

    return model, model_id


# logistic regression model
def logReg(X_train, Y_train, X_test, Y_test):

    model_id = getModelID("logReg")
    logging.info(f"Model ID: {model_id}")

    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)

    score = classifier.score(X_test, Y_test)
    print("Accuracy:", score)

    # predict on train dataset
    Y_train_pred = classifier.predict(X_train)

    # Predict the response for test dataset
    Y_test_pred = classifier.predict(X_test)

    metrics_results = getMetrics(
        Y_train,
        Y_train_pred,
        Y_test,
        Y_test_pred,
        model_id,
    )

    pprint.pprint(metrics_results)
    logging.info(f'accuracy_score_train: {metrics_results["accuracy_score_train"]}')
    logging.info(f'accuracy_score_test: {metrics_results["accuracy_score_test"]}')
    logging.info(f'f1_score_train: {metrics_results["f1_score_train"]}')
    logging.info(f'f1_score_test: {metrics_results["f1_score_test"]}')

    saveMetrics(metrics_results, outputs_path="outputs")

    return classifier, model_id


# linear neural network
def neuralNet(X_train, Y_train, X_test, Y_test):

    model_id = getModelID("neuralNet")
    logging.info(f"Model ID: {model_id}")

    input_dim = X_train.shape[1]

    # define model
    model = Sequential()
    model.add(layers.Dense(10, input_dim=input_dim, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    # compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    # fit model to the training data
    model.fit(
        X_train,
        Y_train,
        epochs=100,
        verbose=False,
        validation_data=(X_test, Y_test),
        batch_size=10,
    )

    # evaluate accuracies for train and test set
    loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))

    # predict on train dataset
    Y_train_pred = model.predict(X_train)

    # Predict the response for test dataset
    Y_test_pred = model.predict(X_test)

    # convert to class
    threshold = 0.5
    Y_train_pred = [1 if x >= threshold else 0 for x in Y_train_pred]
    Y_test_pred = [1 if x >= threshold else 0 for x in Y_test_pred]

    metrics_results = getMetrics(
        Y_train,
        Y_train_pred,
        Y_test,
        Y_test_pred,
        model_id,
    )

    pprint.pprint(metrics_results)
    logging.info(f'accuracy_score_train: {metrics_results["accuracy_score_train"]}')
    logging.info(f'accuracy_score_test: {metrics_results["accuracy_score_test"]}')
    logging.info(f'f1_score_train: {metrics_results["f1_score_train"]}')
    logging.info(f'f1_score_test: {metrics_results["f1_score_test"]}')

    saveMetrics(metrics_results, outputs_path="outputs")

    return model, model_id


# convolutional neural network
def cnn(blogs_train, blogs_test, Y_train, Y_test):

    model_id = getModelID("cnn")
    logging.info(f"Model ID: {model_id}")

    # tokenizer fit on training data
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(blogs_train)

    # blogs are turned to sequenuences by means of the tokenizer
    X_train = tokenizer.texts_to_sequences(blogs_train)
    X_test = tokenizer.texts_to_sequences(blogs_test)

    # save tokenizer
    saveTokenizer(tokenizer, model_id, outputs_path="models")

    vocab_size = len(tokenizer.word_index) + 1

    # the data is padded so as to be of the same length throughout
    maxlen = 32
    X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
    X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)

    embedding_dim = 200

    # model is defined
    model = Sequential()
    model.add(
        layers.Embedding(vocab_size, embedding_dim, input_length=maxlen)
    )  # embeddings for words are created
    model.add(layers.Dropout(0.3))  # dropout used for regularization
    model.add(layers.Conv1D(128, 10, activation="LeakyReLU"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", get_f1]
    )  # compile model
    model.summary()

    # model fit to the training data
    history = model.fit(
        X_train, Y_train, epochs=2, validation_data=(X_test, Y_test), batch_size=10
    )

    # # accuracy for training and test set is evaluated
    # loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
    # print("Training Accuracy: {:.4f}".format(accuracy))
    # loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
    # print("Testing Accuracy:  {:.4f}".format(accuracy))

    # predict on train dataset
    Y_train_pred = model.predict(X_train)
    # Predict the response for test dataset
    Y_test_pred = model.predict(X_test)

    # convert to class
    threshold = 0.5
    Y_train_pred = [1 if x >= threshold else 0 for x in Y_train_pred]
    Y_test_pred = [1 if x >= threshold else 0 for x in Y_test_pred]

    metrics_results = getMetrics(
        Y_train,
        Y_train_pred,
        Y_test,
        Y_test_pred,
        model_id,
    )

    pprint.pprint(metrics_results)
    logging.info(f'accuracy_score_train: {metrics_results["accuracy_score_train"]}')
    logging.info(f'accuracy_score_test: {metrics_results["accuracy_score_test"]}')
    logging.info(f'f1_score_train: {metrics_results["f1_score_train"]}')
    logging.info(f'f1_score_test: {metrics_results["f1_score_test"]}')

    # plot learning curves
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{model_id} model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(os.path.join("outputs", f"{model_id}_training_accuracy.png"))
    plt.close()

    plt.plot(history.history["get_f1"])
    plt.plot(history.history["val_get_f1"])
    plt.title(f"{model_id} model F1 score")
    plt.ylabel("F1 score")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(os.path.join("outputs", f"{model_id}_training_f1_score.png"))
    plt.close()

    saveMetrics(metrics_results, outputs_path="outputs")
    saveModel(model, model_id, outputs_path="models")

    return model, model_id


def lstm(blogs_train, blogs_test, Y_train, Y_test):

    model_id = getModelID("lstm")
    logging.info(f"Model ID: {model_id}")

    # tokenize words
    max_words = 1000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(blogs_train)

    # convert to sequence matrix
    X_train = tokenizer.texts_to_sequences(blogs_train)
    X_test = tokenizer.texts_to_sequences(blogs_test)

    # save tokenizer
    saveTokenizer(tokenizer, model_id, outputs_path="models")

    # the data is padded so as to be of the same length throughout
    maxlen = 32
    X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
    X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)

    embedding_dim = 100
    vocab_size = len(tokenizer.word_index) + 1

    # model is defined
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    # model.add(layers.LSTM(128, activation='relu', return_sequences=True))
    # model.add(layers.Dropout(0.1))
    model.add(layers.LSTM(128, activation="relu"))
    model.add(layers.Dropout(0.2))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", get_f1]
    )
    model.summary()

    # model fit to the training data
    history = model.fit(
        X_train,
        Y_train,
        epochs=10,
        verbose=True,
        validation_data=(X_test, Y_test),
        batch_size=32,
    )

    # plot learning curves
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(f"{model_id} model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(os.path.join("outputs", f"{model_id}_training_accuracy.png"))
    plt.close()

    plt.plot(history.history["get_f1"])
    plt.plot(history.history["val_get_f1"])
    plt.title(f"{model_id} model F1 score")
    plt.ylabel("F1 score")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.savefig(os.path.join("outputs", f"{model_id}_training_f1_score.png"))
    plt.close()

    # predict on train dataset
    Y_train_pred = model.predict(X_train)
    # Predict the response for test dataset
    Y_test_pred = model.predict(X_test)

    # convert to class
    threshold = 0.5
    Y_train_pred = [1 if x >= threshold else 0 for x in Y_train_pred]
    Y_test_pred = [1 if x >= threshold else 0 for x in Y_test_pred]

    metrics_results = getMetrics(
        Y_train,
        Y_train_pred,
        Y_test,
        Y_test_pred,
        model_id,
    )

    pprint.pprint(metrics_results)
    logging.info(f'accuracy_score_train: {metrics_results["accuracy_score_train"]}')
    logging.info(f'accuracy_score_test: {metrics_results["accuracy_score_test"]}')
    logging.info(f'f1_score_train: {metrics_results["f1_score_train"]}')
    logging.info(f'f1_score_test: {metrics_results["f1_score_test"]}')

    saveMetrics(metrics_results, outputs_path="outputs")
    saveModel(model, model_id, outputs_path="models")

    return model, model_id


def vectorize(texts):
    # vectorizer so as to turn the words to number values
    vectorizer = TfidfVectorizer(
        lowercase=True,
        min_df=0.1,  # min document frequency of term - removes low freq words
        ngram_range=(1, 3),
        stop_words="english",
    )
    vectorizer.fit(texts)

    return vectorizer


def main():

    logging.basicConfig(
        filename=os.path.join("outputs", "training.log"),
        encoding="utf-8",
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    print(
        "What data should we use?: "
        "\n0) Pre-processed data"
        "\n1) Re-process data"
        "\nInput: "
    )
    choice = input()  # user input

    if choice == "0":
        # load the pre processed data
        print("Loading pre-processed data")
        logging.info("Loading pre-processed data")
        npzFile = np.load(os.path.join("data", "processed", "preProcessed.npz"))
        text = npzFile["text"]
        labels = npzFile["labels"]
    elif choice == "1":
        # load the pre processed data
        print("Generating re-processed data")
        logging.info("Generating re-processed data")
        text, labels = generateData(os.path.join("data", "raw", "train.csv"))
    else:
        logging.info("Invalid data input")
        print("Invalid input")

    # split training and test set
    # keep "_text" variant for CNN later
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        text, labels, test_size=0.2, random_state=1000
    )

    # fit vectorizer only on training data
    vectorizer = vectorize(X_train_text)

    # change data set from words to numeric values
    X_train = vectorizer.transform(X_train_text).toarray()
    X_test = vectorizer.transform(X_val_text).toarray()
    encoder = LabelBinarizer()
    encoder.fit(y_train)
    Y_train = encoder.transform(y_train).ravel()
    Y_test = encoder.transform(y_val).ravel()

    # main menu is displayed
    print(
        "Do you wish to run: "
        "\n0) All"
        "\n1) Naive Bayes Classifier"
        "\n2) Support Vector Machine"
        "\n3) Logistic Regression Model"
        "\n4) Linear Neural Network"
        "\n5) Convolutional Neural Network"
        "\n6) Long-Short term NN (LSTM)"
        "\nInput: "
    )
    choice = input()  # user input

    # according to the user choice, the appropriate function is carried out
    if choice == "0":
        logging.info("Model naiveBayes running")
        naiveBayes(X_train, Y_train, X_test, Y_test)
        logging.info("Model SVM running")
        svmCode(X_train, Y_train, X_test, Y_test)
        logging.info("Model LogisticRegression running")
        logReg(X_train, Y_train, X_test, Y_test)
        logging.info("Model neuralNet running")
        neuralNet(X_train, Y_train, X_test, Y_test)
        logging.info("Model CNN running")
        cnn(X_train_text, X_val_text, Y_train, Y_test)
        logging.info("Model LSTM running")
        lstm(X_train_text, X_val_text, Y_train, Y_test)
    elif choice == "1":
        logging.info("Model naiveBayes running")
        naiveBayes(X_train, Y_train, X_test, Y_test)
    elif choice == "2":
        logging.info("Model SVM running")
        svmCode(X_train, Y_train, X_test, Y_test)
    elif choice == "3":
        logging.info("Model LogisticRegression running")
        logReg(X_train, Y_train, X_test, Y_test)
    elif choice == "4":
        logging.info("Model neuralNet running")
        neuralNet(X_train, Y_train, X_test, Y_test)
    elif choice == "5":
        logging.info("Model CNN running")
        cnn(X_train_text, X_val_text, Y_train, Y_test)
    elif choice == "6":
        logging.info("Model LSTM running")
        lstm(X_train_text, X_val_text, Y_train, Y_test)

    else:
        logging.info("Invalid model input")
        print("Invalid model input")

    logging.info("Done")


if __name__ == "__main__":
    main()
