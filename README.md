# PRE TRAINED MODELS gdrive link
[pre_trained.zip](https://drive.google.com/file/d/1TJtrry4CNEFmLLOlMAIm6WuNvSG07NuV/view?usp=drive_link)

* the above link contains `pre_trained.zip` file please unzip using `unzip -j pre_trained.zip`
* forward_model.pt -> pretrained forward language model embeddings used in ELMO 
* backward_model.pt -> pretrained backward language model embeddings used in ELMO 
* untrainable_lambdas.pt -> Downstream model with untrained lambdas
* trainable_lambdas.pt -> Downstream model with trained lambdas
* trainable_function.pt -> Downstream model with learning function of embeddings
* test_data.pt -> processed data of test.csv
* train_data.pt -> processed data of train.csv

# INSTRUCTIONS

* If gdown is installed running `python download.py` will download and unzip those pre_trained models above.
* `prepare.py` is used for processing the test.csv and train.csv respectively to generate corpus.
* `datautils.py` is used for implementing cleaning and tokenization of sentences in the corpus.
*  `ELMO.py` used to generate the forward and backward LSTM which are used to  generate embeddings by running `python ELMO.py`.
* `classification.py` used to perform downstream classifcation  using ELMO by running `python classification.py`.
* `evaluate.py` used to generate evaluation metrics for all the models by running `python evaluate.py`.

