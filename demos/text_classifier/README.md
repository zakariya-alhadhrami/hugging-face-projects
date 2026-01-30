

# 🍓 Food Not Food Classifier

## 📋 Project Metadata

| Field       | Value                    |
| ----------- | ------------------------ |
| Title       | Food Not Food Classifier |
| Emoji       | 🍓                       |
| Color From  | gray                     |
| Color To    | red                      |
| SDK         | Gradio                   |
| SDK Version | 4.40.0                   |
| App File    | app.py                   |
| Pinned      | false                    |
| License     | Apache-2.0               |

A simple, clean, and interactive **NLP web app** that predicts whether a given sentence is **food-related** or **not food-related**.

The app is built with **Gradio** and powered by a **fine-tuned DistilBERT** model, making it fast, lightweight, and easy to use.

---

## ✨ Features

* 🔍 Classifies short text into **Food** or **Not Food**
* ⚡ Fast inference using **DistilBERT**
* 🎛️ Clean and interactive **Gradio UI**
* 🧠 Trained on synthetic food / non-food captions

---

## 🧠 Model

* **Base model:** DistilBERT
* **Task:** Binary text classification
* **Labels:** `food`, `not_food`
* **Training data:** ~250 synthetically generated text captions

The model focuses on learning **semantic meaning**, not just keyword matching.

---

## 📊 Dataset

* Synthetic dataset created for experimentation and learning
* Balanced between food-related and non-food-related sentences
* Examples include:

  * Food: *"A plate of pasta with tomato sauce"*
  * Not Food: *"A laptop sitting on a wooden desk"*

> ⚠️ Note: Because the dataset is small and synthetic, predictions may not generalize perfectly to all real-world text.

---

## 🚀 How to Use

1. Enter a sentence in the text box
2. Click **Submit**
3. View the predicted label and confidence score

---

## 🛠️ Tech Stack

* Python
* Hugging Face Transformers
* PyTorch
* Gradio

---

## 📦 Running Locally

```bash
pip install -r requirements.txt
python app.py
```

Then open the provided local Gradio URL in your browser.

---

## 📌 Limitations

* Small training dataset
* English-only text
* Not optimized for long paragraphs

---

## 📄 License

This project is licensed under the **Apache 2.0 License**.

---

## 🙌 Acknowledgements

* Hugging Face 🤗
* Gradio

Feel free to fork, experiment, and improve the model!


