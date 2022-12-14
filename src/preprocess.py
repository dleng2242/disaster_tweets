import re
import os
import contractions
import inflect
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# Define what is to be replaced
REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_]")
STOPWORDS = set(stopwords.words("english"))


def alterations(words):
    """
    Performs stemming, lemming, and conversion of numbers to alphabetical form.
    """
    p = inflect.engine()
    sno = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    for word in words.split():
        new_word = word
        # replaces numbers in numeric form with their alphabetic form
        if word.isnumeric():
            try:
                new_word = p.number_to_words(word)
                new_word = new_word.replace(",", "")
            except:
                new_word = "one"
        stem = sno.stem(new_word)  # stems words to their root
        lemma = lemmatizer.lemmatize(stem, pos="v")  # lemetizes words
        words = words.replace(
            " " + word + " ", " " + lemma + " "
        )  # replaces the old word with the altered version

    return words


def clean_text(text):
    """
    Cleans string.

    Converts to lower case, converts non-ascii to ascii, fixes contraction
    replaces URLs and ellipsis with text, removes/cleans symbols, removes
    stop words and applies the stemming and lemming.

    """
    text = text.lower()  # lowercase text
    text = clean_non_ascii(text)
    text = contractions.fix(text)  # Replace contractions in string of text
    text = replace_URL(text)
    text = replace_ellipsis(text)
    text = REPLACE_BY_SPACE_RE.sub(
        " ", text
    )  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(
        "", text
    )  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = replace_symbol(text, "#")
    text = replace_symbol(text, "@")
    text = " ".join(
        word for word in text.split() if word not in STOPWORDS
    )  # delete stopwords from text
    text = alterations(text)  # applies alterations to text

    return text


def replace_URL(text):
    t = re.sub(r"http\S+", "alinkwashere", text)
    return t


def replace_symbol(text, symbol):
    new_symbol = symbol + " "
    t = re.sub(symbol, new_symbol, text)
    return t


def replace_ellipsis(text):
    t = re.sub("\...", " ellipsiswashere ", text)
    t = re.sub("  ", " ", t)
    return t


def clean_non_ascii(text):
    """Removes non-ascii '\x89' or 'åÈ' style characters"""
    text = text.encode("ascii", "ignore").decode("ascii")
    return text


def remove_duplicate_text(text, labels):
    res = {text[i]: labels[i] for i in range(len(text))}
    return (list(res.keys()), list(res.values()))


def generateData(filename, dataset_type="train"):

    # Read the CSV file
    data = pd.read_csv(filename)

    # form a list of texts
    text = data.loc[:, "text"].tolist()
    cleanText = []

    # clean the text
    for line in text:
        cleanText.append(clean_text(line))

    if dataset_type == "train":
        # get lost of labels
        labels = data.loc[:, "target"].tolist()
        # deduplicate
        cleanText, labels = remove_duplicate_text(cleanText, labels)

        return cleanText, labels
    if dataset_type == "test":
        return cleanText
    else:
        print("Please choose either 'train' or 'test'")


def main():
    # returns the properly formatted data
    text, labels = generateData(os.path.join("data", "raw", "train.csv"))
    # saves the arrays to a file to be read from later
    file_path = os.path.join("data", "processed", "preProcessed.npz")
    np.savez(file_path, text=text, labels=labels)
    print("Processed data saved: ", file_path)


if __name__ == "__main__":
    main()
