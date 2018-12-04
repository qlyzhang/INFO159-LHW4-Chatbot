# INFO159-LHW4-Chatbot

**Group member:** Changxu Zhang & You Zhang

**Github link:** [https://github.com/qlyzhang/INFO159-LHW4-Chatbot](https://github.com/qlyzhang/INFO159-LHW4-Chatbot)

**Model link:** [https://drive.google.com/file/d/1qVZuQTbLPXZgrjNF09VjSCyPCfwxsgtE/view?usp=sharing](https://drive.google.com/file/d/1qVZuQTbLPXZgrjNF09VjSCyPCfwxsgtE/view?usp=sharing)

This is a pytorch seq2seq Chatbot for [INFO159 Natural Language Processing in UC Berkeley](https://http://people.ischool.berkeley.edu/~dbamman/nlp18.html) 

We implemented [A Neural Conversational Model](https://arxiv.org/pdf/1506.05869.pdf) which is a sequence-to-sequence model introduced in [lecture23](http://people.ischool.berkeley.edu/~dbamman/nlpF18/slides/23_dialogue.pdf)

We trained our chatbot on NVIDIA GTX 1070 with data  `movie_lines.tsv` from the [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

## Requirements
* python 3
* pytorch 0.4.0
* numpy
* pandas

## Referrence

* [https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
* [https://github.com/Conchylicultor/DeepQA/](https://github.com/Conchylicultor/DeepQA/)
* [https://pytorch.org/tutorials/beginner/chatbot_tutorial.html](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
* [https://github.com/Currie32/Chatbot-from-Movie-Dialogue/](https://github.com/Currie32/Chatbot-from-Movie-Dialogue/)

## How to run the code
* run the code 
```
python3 LHW4_Seq2Seq_Chatbot.py
```

<!--
* If you want to use our pretrained model:
```
python3 LHW4_Seq2Seq_Chatbot.py --pretrained
```
-->

* See the jupyter notebook [here](https://github.com/qlyzhang/INFO159-LHW4-Chatbot/blob/master/LHW4_Seq2Seq_Chatbot.ipynb)


## What we do
### Data Pre-processing
First, we do some data cleaning and remove some weird symbols. Then, we sort the data according to `movieID` and `lineID`.We reorganized the data into pairs of QandAs and remove the pair if there are infrequent words in it.
#### Read Data
A dataframe is created to organize the data. We separate data into 5 columns: `lineID`,  `characterID`,  `movieID`,   `character name`,  `text of the utterance`
#### Clean Data
Use regular expressions to replace some weird strings into space.
#### Sort Data
Sort the data according to `movieID` and `lineID`
#### Load Q&A Data
Reorganized the `text of the utterance` data into pairs of Q&As
#### Count Data
Count how many times each word occurs in our clean data
#### Trim and Index Data
Trim the Q&A pair if there are infrequent words in it. Make two dictionaries `word2index` and `index2word`
#### Prepare Data for Model
Convert the dictionaries into tensors with PyTorch to prepare for model.

### Model
Use an Encoder-Decoder Recurrent Neural Network Model.

## Our Result

### Some good result
Human > hello

Bot > hi

Human > How are you ?

Bot > fine

Human > Who are you ?

Bot > i m not sure.

Human > that is cool

Bot > you know what i m doing here ?

Human > you are under arrest

Bot > i m not .

Human > hjvksbv asdkb

Bot > Sorry, I don't know what you mean

### Some unexpected result
Human > Are you my friend?

Bot > i m fine

Human > i don't know

Bot > you re a good time .

Human > Are you human?

Bot > yes .
