Notebooks from the Seminars:

# Einführung in die Programmierung Künstlicher Intelligenzen  < WS18/19 - WS19/20

### Introduction to Artificial Intelligence Programming

Georg Trogemann und Christian Heck

Grundlagenseminar Material/Skulptur/Code

Dienstag wöchentlich 11:00 –13:00

Filzengraben 8 - 10, 0.2 [Experimentelle Informatik](https://www.khm.de/exMedia_experimentelle_informatik/)

Kunsthochschule für Medien Köln

Email: g.trogemann@khm.de, c.heck@khm.de

### Description

Profound cultural consequences of AI do not only appear with the use of upload filters for algorithmic censorship of undesirable text and image content or the auctioning of AI paintings at Christie's; nor with the formulation of ethical guidelines for dealing with AI or the increased emergence of AI-powered hate speech bots. They begin, quite abstractly and mostly unnoticed in their programming, in semi-public, very formal fields of discourse.

This is exactly where we start experimentally. The seminar provides a very elementary introduction to the subsymbolic AI of neural networks and their programming. The aim of this seminar is to code from scratch, discuss the code together and learn to understand it, in order to learn to assess the possibilities, limits and dangers of this technology for oneself.

We do not adopt the technology of artificial intelligence as a tool in the Homo Faberian sense, but combine programming as an artistic practice with the critical analysis of its social effects.

### Info 

**Seminar Wiki Pages:**

- [WS18-19](https://exmediawiki.khm.de/exmediawiki/index.php/Einf%C3%BChrung_in_die_Programmierung_k%C3%BCnstlicher_Intelligenzen)
- [WS19-20](https://exmediawiki.khm.de/exmediawiki/index.php/AI@exLabIII)

**Executing the Notebooks:**

- *You can run, execute and work on the following Notebooks here:* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/experimental-informatics/hands-on-artificial-neural-networks/HEAD)

### Setting Up

**Basics in Anaconda & Jupyter Notebooks:**

* [KI_Workaround_installieren](https://exmediawiki.khm.de/exmediawiki/index.php/KI_Workaround_installieren)

---

### [Hands on Python](https://github.com/experimental-informatics/hands-on-python)

see repository: https://github.com/experimental-informatics/hands-on-python

---

### [Artificial Neural Net in Python](./02_ANN-in-Python)

many of the Code is based on Tariq Rashid's Book [»Neuronale Netze selbst programmieren«](https://www.oreilly.com/library/view/neuronale-netze-selbst/9781492064046/) < [Git Repo]( https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork)

* Coding a **Dense Neural Net** in Python **from Scratch**:
  * [KNN_in-Python_am-eigenen-Bild.ipynb](./02_ANN-in-Python/KNN_in-Python_am-eigenen-Bild.ipynb)

* **Data Preprocessing:**
  * [Preprocessing.ipynb](./02_ANN-in-Python/Preprocessing.ipynb)

---

### [Hands on Keras & Tensorflow](./03_Hands-on-Keras)

all examples working with MNIST handwritten digit database

most of the Codes are based on Francois Chollet's Book [»Deep Learning with Python«](https://www.manning.com/books/deep-learning-with-python) < [Git Repo]( https://github.com/fchollet/deep-learning-with-python-notebooks)

##### Dense Neural Net

* Coding a**Artificial Neural Net** (DNN) in Tensorflow & Keras:
  * [ANN-in-Keras.ipynb](./03_Hands-on-Keras/ANN-in-Keras.ipynb)

##### Autoencoder

* Simple **Autoencoder** in Tensorflow & Keras
  * [Autoencoder-in-Keras.ipynb](./03_Hands-on-Keras/Autoencoder-in-Keras.ipynb)

##### Convolutional Neural Net

* **CNN** in Tensorflow & Keras
  * [CNN-in-Keras.ipynb](./03_Hands-on-Keras/CNN-in-Keras.ipynb)

##### Generative Adversarial Network

* **GAN** in Tensorflow & Keras
  * [GAN-in-Keras.ipynb](./03_Hands-on-Keras/GAN-in-Keras.ipynb)

##### [Interpretable AI](./04_InterpretableAI)

* **Visualize Activations** (based on model from [CNN-in-Keras.ipynb](./03_Hands-on-Keras/CNN-in-Keras.ipynb))
  * [mnist_cnn-activations_keract.ipynb](./04_InterpretableAI/mnist_cnn-activations_keract.ipynb)

* **Visualize Heatmaps** (based on model from [CNN-in-Keras.ipynb](./03_Hands-on-Keras/CNN-in-Keras.ipynb))
  * [mnist_cnn-heatmaps_keract.ipynb](./04_InterpretableAI/mnist_cnn-heatmaps_keract.ipynb)

##### [Explainable AI](./05_ExplainableAI)

* **LIME** for image classification by using Keras (InceptionV3)
  * [kiss-explained_lime-demo.ipynb](./05_ExplainableAI/kiss-explained_lime-demo.ipynb)

---

### [Natural Language Processing (NLP)](./06_NLP)

##### Text Preprocessing

* for english Textcorpora
  * [tokenize_english_textcorpora.ipynb](./06_NLP/tokenize_english_textcorpora.ipynb)

* for german Textcorpora
  * [tokenize_german_textcorpora.ipynb](./06_NLP/tokenize_german_textcorpora.ipynb)

##### Chatbots

* Basic Encodings & traditional embeddings (ONE-HOT / BOW / TF-IDF)
  * [one-hot_bow_tfidf.ipynb](./06_NLP/one-hot_bow_tfidf.ipynb)
* Basic Chatbots (TF-IDF)
  * [simple_german_chatbot.ipynb](./06_NLP/simple_german_chatbot.ipynb)
  * [simple_english_chatbot.ipynb](./06_NLP/simple_english_chatbot.ipynb)
* Chatbots with Chatterbot
  * [chatterbot.ipynb](./06_NLP/chatterbot.ipynb)

##### Sentiment Analysis

* Sentiment Analysis for german textcorpora
  * [Sentiment-analyses_4_german-texts_with_TextBlob.ipynb](./06_NLP/Sentiment-analyses_4_german-texts_with_TextBlob.ipynb)

##### Word embeddings

* train a Word2vev Modell on your own Texts
  * [train_your_own_Word2vec-model.ipynb](./06_NLP/train_your_own_Word2vec-model.ipynb)

##### RNN/LSTM

* **LSTM**- Textgeneration
  * [LSTM-Textgenerator.ipynb](./06_NLP/LSTM-Textgenerator.ipynb)

##### Datasets

* Loading and scraping data from the web
  * [load_scrape_data-sets.ipynb](./06_NLP/load_scrape_data-sets.ipynb)

---

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/experimental-informatics/hands-on-artificial-neural-networks/HEAD)
