
# Disaster Tweets Classification

My attempt at the Kaggle NLP disaster tweet classification problem,
found [here](https://www.kaggle.com/competitions/nlp-getting-started/overview). 

This work was originally developed during an internal mini-hackathon
where I worked over two days as part of a small team. I have since 
tidied and re-organised the code. 

## Quick start

Use the `environment.yml` to recreate the conda environment as usual. 

Within the `src/` folder there are the three scripts used to pre-process the data,
`preprocessing.py`, 
run the training `run_training.py`, and run the inference `run_inference.py`. 

Make sure you have the training and testing data in `data/raw/`. 
Running the processing script will generate the pre-processed data and save it 
in `data/processed/`. 

Run the training script to train a model, using the interactive prompt to 
choose which model to train and if you want to re-process the data. 

Once you have generated a trained model, run the inference script to generate
predictions using the model ID generated during training.

Training logs and model summary metrics are saved to `outputs/`. 

## Process 

We initially conducted an EDA and then set about building a pre-processing script
and a training script with a set of models. 

The pre-processing script did a lot of standard text cleaning such as stripping 
special characters, dealing with encoding issues, removing stop words,
and stemming and lemming.
It was also observed that tweets with links in were more likely to be disaster
related and so we also pulled these out as features. 

When building models we started by bench marking against
some classical ML models for classification such as Naive Bayes, SVM, and 
logistic regression, and then went on to build a simple
neural network, and two more complex neural networks: a CNN and an LSTM.
Logging was set up to track models and metrics during training. 
Due to the time and computational constraint of the hackathon we could only 
trail a limited set of models and hyperparameters. 

An inference script was written to generate predictions from any exported
model (with tokenizer). 

The best performing models were the CNN and LSTM and so these predictions 
were submitted to Kaggle achieving an F1 score of 0.770 and 0.776 respectively.

If we had more time we would have loved to have done some error analysis
to see what was/was not working. I would have also wanted to tried out various
levels of cleaning and featurization.

In future, using some auto-ml solution like [auto-keras](https://autokeras.com/) 
or [AutoGluon](https://auto.gluon.ai/stable/index.html) would probably lead to
a higher score. Or at least spending time automating the hyperparameter tuning. 
I would also like to try applying a pre-trained model like BERT. 

