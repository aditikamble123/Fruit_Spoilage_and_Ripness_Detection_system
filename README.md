# 🍎 Fruit Ripeness Detection System

An AI-powered fruit ripeness detection system that classifies fruits into different ripeness stages using image processing and deep learning. The system is designed to support real-time prediction and can be extended for smart agriculture and food quality monitoring applications.

---

## 🚀 Features

* 🍏 Image-based fruit ripeness classification
* 🧠 CNN-based deep learning model
* ⚡ Real-time prediction capability
* 🏷️ Automated dataset labeling & preprocessing
* 📈 Training performance visualization (accuracy & loss)

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Machine Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV
* **Data Processing:** NumPy, Pandas
* **Development:** Jupyter Notebook

---

## 📂 Project Structure

```
Fruit_Ripeness/
│
├── Train/                     # Training dataset
├── Test/                      # Testing dataset
├── models/                    # Saved trained models
│
├── train.py                   # Model training script
├── predict.py                 # Image prediction script
├── realtime.py                # Real-time detection
├── generate_labels.py         # Dataset labeling
│
├── fruit_training_notebook.ipynb   # Training workflow
├── training_history.png            # Accuracy & loss graph
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/aditikamble123/Fruit_Ripeness.git
cd Fruit_Ripeness
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the model

```bash
python train.py
```

### Run prediction

```bash
python predict.py
```

### Real-time detection

```bash
python realtime.py
```

---

## 📊 Results

* Model performance is visualized using accuracy and loss curves
<img width="4470" height="2970" alt="training_history" src="https://github.com/user-attachments/assets/f3c10b7f-8ea4-4e7b-9e47-ede945f1df04" />

---

## 🌍 Applications

* Smart agriculture & precision farming
* Automated fruit quality inspection
* Supply chain & storage monitoring
* Food waste reduction systems

---

## 🔐 Note

Sensitive files such as API keys (e.g., Firebase credentials) are not included in this repository for security reasons.

---

## 👩‍💻 Author

**Aditi K**
B.Tech – Industrial IoT

---

## ⭐ Future Improvements

* Integration with IoT devices (ESP32 / sensors)
* Mobile or web-based interface
* Multi-fruit and multi-class classification
* Cloud deployment for scalability

---
