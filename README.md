Notebooks from the Seminars:

# Einführung in die Programmierung Künstlicher Intelligenzen  < WS18/19 - WS19/20

Georg Trogemann und Christian Heck

Grundlagenseminar Material/Skulptur/Code

Dienstag wöchentlich 11:00 –13:00

Filzengraben 8 - 10, 0.2 [Experimentelle Informatik](https://www.khm.de/exMedia_experimentelle_informatik/)

Kunsthochschule für Medien Köln

Email: g.trogemann@khm.de, c.heck@khm.de

## Description

Tief greifende kulturelle Konsequenzen von KI treten nicht erst beim Einsatz von Uploadfiltern zur algorithmischen Zensur unerwünschter Text- und Bildinhalte oder der Versteigerung von KI-Gemälden bei Christie's in Erscheinung; auch nicht bei der Ausformulierung ethischer Leitlinien für den Umgang mit KI oder dem vermehrten Aufkommen von AI powered Hate Speech Bots. Sie beginnen, ganz abstrakt und meist unbeachtet bei ihrer Programmierung, in semi-öffentlich geführten, sehr formalen Diskursfeldern.

Genau dort setzen wir experimentell an. Das Seminar führt sehr elementar in die subsymbolische KI der Neuronalen Netze und deren Programmierung ein. Coden from scratch, den Code gemeinsam diskutieren und zu verstehen lernen, um auf diesem Wege die Möglichkeiten, Grenzen und Gefahren dieser Technologie für sich einschätzen zu lernen ist Ziel dieses Seminars.

Wir machen uns die Technologie der Künstlichen Intelligenz nicht als ein Tool im Homo Faberischen Sinne zu eigen, sondern verbinden Programmieren als künstlerische Praxis mit der kritischen Analyse ihrer gesellschaftlichen Auswirkungen.

## Info 

**Seminar Wiki Pages:**

- [WS18-19](https://exmediawiki.khm.de/exmediawiki/index.php/Einf%C3%BChrung_in_die_Programmierung_k%C3%BCnstlicher_Intelligenzen)
- [WS19-20](https://exmediawiki.khm.de/exmediawiki/index.php/AI@exLabIII)

**Executing the Notebooks:**

- *You can run, execute and work on the following Notebooks here:* [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/experimental-informatics/hands-on-artificial-neural-networks/HEAD)

## Setting Up

**Basics in Anaconda & Jupyter Notebooks:**

* [KI_Workaround_installieren](https://exmediawiki.khm.de/exmediawiki/index.php/KI_Workaround_installieren)

---

## [Hands on Python](https://github.com/experimental-informatics/hands-on-python)

see repository: https://github.com/experimental-informatics/hands-on-python

---

## [Artificial Neural Net in Python](./02_ANN-in-Python)

many of the Code is based on Tariq Rashid's Book [»Neuronale Netze selbst programmieren«](https://www.oreilly.com/library/view/neuronale-netze-selbst/9781492064046/) < [Git Repo]( https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork)

* Coding a **Dense Neural Net** in Python **from Scratch**:
  * [KNN_in-Python_am-eigenen-Bild.ipynb](./02_ANN-in-Python/KNN_in-Python_am-eigenen-Bild.ipynb)

* **Data Preprocessing:**
  * [Preprocessing.ipynb](./02_ANN-in-Python/Preprocessing.ipynb)

---

## [Hands on Keras & Tensorflow](./03_Hands-on-Keras)

all examples working with MNIST handwritten digit database

most of the Codes are based on Francois Chollet's Book [»Deep Learning with Python«](https://www.manning.com/books/deep-learning-with-python) < [Git Repo]( https://github.com/fchollet/deep-learning-with-python-notebooks)

### Dense Neural Net

* **Artificial Neural Net** (DNN) in Tensorflow & Keras:
  * [ANN-in-Keras.ipynb](./03_Hands-on-Keras/ANN-in-Keras.ipynb)

### Autoencoder

* Simple **Autoencoder** in Tensorflow & Keras
  * [Autoencoder-in-Keras.ipynb](./03_Hands-on-Keras/Autoencoder-in-Keras.ipynb)

### Convolutional Neural Net

* **CNN** in Tensorflow & Keras
  * [CNN-in-Keras.ipynb](./03_Hands-on-Keras/CNN-in-Keras.ipynb)

### Generative Adversarial Network

* **GAN** in Tensorflow & Keras
  * [GAN-in-Keras.ipynb](./03_Hands-on-Keras/GAN-in-Keras.ipynb)

---

### [Interpretable AI](./04_InterpretableAI)

* **Visualize Activations** (based on model from [CNN-in-Keras.ipynb](./03_Hands-on-Keras/CNN-in-Keras.ipynb))
  * [mnist_cnn-activations_keract.ipynb](./04_InterpretableAI/mnist_cnn-activations_keract.ipynb)

* **Visualize Heatmaps** (based on model from [CNN-in-Keras.ipynb](./03_Hands-on-Keras/CNN-in-Keras.ipynb))
  * [mnist_cnn-heatmaps_keract.ipynb](./04_InterpretableAI/mnist_cnn-heatmaps_keract.ipynb)

---

### [Explainable AI](./05_ExplainableAI)

* **LIME** for image classification by using Keras (InceptionV3)
  * [kiss-explained_lime-demo.ipynb](./05_ExplainableAI/kiss-explained_lime-demo.ipynb)

---

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/experimental-informatics/hands-on-artificial-neural-networks/HEAD)

