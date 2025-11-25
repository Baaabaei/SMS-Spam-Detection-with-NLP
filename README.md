# SMS Spam Detection with NLP

A comprehensive Natural Language Processing project comparing different approaches for SMS spam classification, including traditional machine learning with TF-IDF and deep learning with LSTM networks using both custom and pre-trained word embeddings.

## 📊 Dataset

- **Source**: SMS Spam Collection Dataset from Kaggle
- **Size**: 5,572 SMS messages
- **Classes**: Binary classification (ham=0, spam=1)
- **Distribution**: 
  - Ham (legitimate): 4,825 messages (~86.6%)
  - Spam: 747 messages (~13.4%)

## 🔍 Text Preprocessing Pipeline

### 1. Data Cleaning
- Removal of unnecessary columns
- Column renaming for clarity

### 2. Text Processing Steps
```python
# Punctuation Removal
# Tokenization (using regex)
# Stopword Removal (NLTK English stopwords)
```

### 3. Advanced Processing
- **Tokenization**: Using Keras Tokenizer (max_words=10,000)
- **Padding**: Sequences padded to uniform length
- **Word Embeddings**: 
  - Custom Word2Vec (trained on dataset)
  - Pre-trained GloVe embeddings (glove-wiki-gigaword-100)

## 🚀 Models Implemented

### 1. Basic LSTM (Without Pre-trained Embeddings)
- **Architecture**:
  - Embedding layer (10,000 words, 50 dimensions)
  - LSTM layer (32 units)
  - Dense layer (32 units, ReLU activation)
  - Output layer (1 unit, Sigmoid activation)
- **Training**: 10 epochs, batch size 32
- **Results**:
  - Final Validation Accuracy: **99.19%**
  - Precision: 99.22%
  - Recall: 94.07%

### 2. LSTM with Custom Word2Vec Embeddings
- **Word2Vec Parameters**:
  - Vector size: 100
  - Window: 5
  - Min count: 2
- **Architecture**: Same as basic LSTM but with pre-trained embeddings (frozen)
- **Results**:
  - Final Validation Accuracy: **96.32%**
  - Precision: 86.15%
  - Recall: 82.96%

### 3. LSTM with GloVe Pre-trained Embeddings
- **GloVe Model**: glove-wiki-gigaword-100
- **Architecture**: LSTM with 100-dimensional GloVe embeddings (frozen)
- **Purpose**: Leverage general language understanding from Wikipedia and Gigaword corpus

## 📈 Key Features

- **Multiple Preprocessing Techniques**: From basic punctuation removal to advanced tokenization
- **Word Embedding Comparison**: Custom Word2Vec vs. Pre-trained GloVe
- **Deep Learning with LSTM**: Sequential modeling for context understanding
- **Comprehensive Metrics**: Accuracy, Precision, and Recall tracking

## 🛠️ Technologies Used

```python
# Core Libraries
numpy
pandas
scikit-learn

# NLP Libraries
nltk
gensim

# Deep Learning
tensorflow
keras

# Preprocessing
re (regex)
string
```

## 🔧 Installation

```bash
# Install required packages
pip install numpy pandas scikit-learn
pip install nltk gensim
pip install tensorflow keras

# Download NLTK stopwords
import nltk
nltk.download('stopwords')
```

## 📝 Usage

1. **Data Loading**:
```python
data = pd.read_csv("spam.csv", encoding='latin-1')
```

2. **Text Preprocessing**:
```python
# Remove punctuation
data["no punc"] = data['txt'].apply(lambda x: remove_punc(x))

# Tokenize
data["tokenized"] = data["no punc"].apply(lambda x: tokenize(x.lower()))

# Remove stopwords
data["no_stop"] = data["tokenized"].apply(lambda x: remove_stopwords(x))
```

3. **Model Training**:
```python
# Prepare sequences
tokenizer = Tokenizer(num_words=10000)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_seq)

# Train LSTM
model.fit(X_train_padded, y_train, 
          batch_size=32, epochs=10,
          validation_data=(X_test_padded, y_test))
```

## 📊 Model Comparison

| Model | Validation Accuracy | Precision | Recall |
|-------|-------------------|-----------|--------|
| Basic LSTM | **99.19%** | 99.22% | 94.07% |
| LSTM + Word2Vec | 96.32% | 86.15% | 82.96% |
| LSTM + GloVe | Testing... | - | - |

## 🎯 Key Insights

1. **Basic LSTM** achieved the highest performance, suggesting that task-specific embeddings learned from scratch can be very effective for this dataset
2. **Custom Word2Vec** provided decent results but with lower precision, indicating potential overfitting to training vocabula
