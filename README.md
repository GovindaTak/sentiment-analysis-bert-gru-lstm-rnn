# sentiment-analysis-bert-gru-lstm-rnn
A comparative study on sentiment analysis using BERT, RNN, LSTM, and GRU. The project evaluates the effectiveness of transformer-based models versus recurrent architectures in classifying tweet sentiments as positive, negative, or neutral.
----
## **Overview**  
This project performs a **comparative analysis of sentiment classification** using different deep learning architectures, including:  
✅ **BERT (Bidirectional Encoder Representations from Transformers)**  
✅ **RNN (Recurrent Neural Network)**  
✅ **LSTM (Long Short-Term Memory)**  
✅ **GRU (Gated Recurrent Unit)**  

The goal is to **evaluate and compare** how these models handle sentiment analysis tasks using a labeled dataset containing user tweets and their corresponding sentiment labels.  

---

## **1️⃣ Dataset Details**  
The dataset consists of tweets labeled as:  
- **0** = Negative Sentiment  
- **1** = Neutral Sentiment  
- **2** = Positive Sentiment  

Each tweet has attributes like:  
✔ **Tweet ID**  
✔ **User Text**  
✔ **Sentiment Label**  
✔ **Time of Tweet**  
✔ **Age of User**  

---

## **2️⃣ Text Preprocessing**  
To clean and standardize the text, the following NLP preprocessing steps were applied:  
✔ **Lowercasing** for consistency  
✔ **Removing URLs, punctuation, and special characters**  
✔ **Stopword Removal** (e.g., "the", "is", "and")  
✔ **Lemmatization** using Spacy  
✔ **Tokenization** (Different for BERT vs. RNN-based models)  

---

## **3️⃣ Model Architectures & Implementation**  

### **A. Transformer-based Model (BERT)**  
BERT was fine-tuned for sequence classification:  
- **Pretrained Model**: `bert-base-uncased`  
- **Optimizer**: AdamW  
- **Loss Function**: CrossEntropyLoss  
- **Batch Size**: 32  
- **Max Sequence Length**: 33  

🚀 **BERT Achieved Accuracy: 77.84%**  

---

### **B. RNN-based Models (RNN, LSTM, GRU)**  
Implemented RNN, LSTM, and GRU models with **100-dimensional Word2Vec embeddings** trained on the dataset.  

**Model Hyperparameters:**  
- **Hidden Units**: 64  
- **Dropout**: 0.5  
- **Optimizer**: Adam  
- **Loss Function**: CrossEntropyLoss  
- **Batch Size**: 32  

🔥 **Results & Comparison**:  

| Model  | Accuracy  | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **BERT**  | 77.84%  | 0.78  | 0.78  | 0.78  |
| **RNN**   | 40.41%  | 0.52  | 0.33  | 0.19  |
| **LSTM**  | 40.46%  | 0.16  | 0.40  | 0.23  |
| **GRU**   | 40.46%  | 0.13  | 0.33  | 0.19  |

📌 **Key Takeaways**:  
✅ **BERT significantly outperforms RNN-based models** in sentiment analysis.  
✅ **RNN, LSTM, and GRU models struggle with classification**, achieving ~40% accuracy.  
✅ **Precision and recall for RNN-based models are lower**, indicating difficulty in distinguishing between sentiment classes.  

---

## **4️⃣ Training & Evaluation**  
###  Fine-Tuning BERT**  

## **5️⃣ Future Enhancements**  
🔹 Implement **BERT Large** for better contextual understanding.  
🔹 Experiment with **Transformer-based architectures like RoBERTa & T5**.  
🔹 Use **Data Augmentation** techniques for class imbalance issues.  

---

## **📌 How to Use**  
1️⃣ **Clone the Repository**  
git clone https://github.com/GovindaTak/sentiment-analysis-bert-gru-lstm-rnn.git
cd sentiment-analysis-bert-gru-lstm-rnn

2️⃣ **Install Dependencies**  

pip install -r requirements.txt

3️⃣ **Run Training**  

python train_model.py

4️⃣ **Evaluate Model Performance**  

python evaluate_model.py

---

## **📜 Summary**  
✔ **BERT outperforms traditional RNN architectures in sentiment analysis.**  
✔ **GRU & LSTM perform slightly better than standard RNN but still struggle with long-term dependencies.**  
✔ **Transformer models like BERT capture better word relationships and outperform recurrent models.**  

---

## **📌 Contact**  
For any queries or collaborations, reach out at **govindatak19@gmail.com**  
Explore my other projects: **[Govinda Tak GitHub](https://github.com/GovindaTak)**  

---

### **🚀 Final Recommendation:**  
If **computational resources are available**, **BERT** should be used for sentiment analysis. Otherwise, **LSTM/GRU** may provide an acceptable balance between accuracy and efficiency.  

🔥 **Let’s push the boundaries of NLP together!** 🚀
