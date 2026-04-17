# Early Misinformation Detection

A simple fake news detection system built using Python. You can give it any text or claim, and it will predict whether it is **real or fake**, along with a confidence score.

The goal of this project is to understand how different techniques can be combined to detect misinformation, instead of relying on just one model.

One important part of this project is that everything is implemented from scratch using NumPy, without using deep learning frameworks like PyTorch or TensorFlow.

---

## How it works

The model does not depend on a single method. Instead, it combines multiple ideas to analyze the input text.

First, the input is checked against a knowledge base using a retrieval-based approach (RAG). If the system finds strong similarity with known data, it makes a quick prediction.

If the match is not strong enough, the text is processed further using different feature extractors:
- Pattern-based features (similar to CNN)
- Sequence-based features (similar to RNN)
- Structure-based features (similar to RVNN)

All these features are combined and passed into a final classifier, which gives the prediction.

This approach helps keep the system both efficient and reasonably accurate.

---

## How to run

Install dependencies:

```
pip install -r requirements.txt
```

Train the model:

```
python src/train.py
```

Run prediction:

```
python src/predict.py --text "your claim here"
```

Evaluate the model:

```
python src/evaluate.py
```

---

## Techniques used

- **TF-IDF**  
  Used to convert text into numerical form.

- **Cosine Similarity**  
  Helps in matching input text with the knowledge base.

- **RAG (Retrieval-Augmented Generation)**  
  Used for quick lookup-based predictions.

- **CNN**  
  Captures local patterns like sensational words.

- **RNN**  
  Captures sequence and flow of text.

- **RVNN**  
  Captures structure and syntactic signals.

- **MLP Classifier**  
  Final layer that decides whether the text is real or fake.

---



---

## Output

The model returns:
- Prediction (Real / Fake)
- Probability score
- Routing information (whether it used retrieval or full model)

---

## Notes

- This project is mainly for learning and experimentation.
- The knowledge base is limited, so results may vary on unseen topics.
- The model can be improved further with better data and tuning.

---

## Team Members

- Parth Singhal  
- Ghanisht Kaushal  
- Rahul  
- Amarveer Singh  
- Angad Singh  
