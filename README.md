# **CIFAR-10 CNN and Flask Deployment**

This project demonstrates the development and deployment of an image classification system using the **CIFAR-10 dataset**, a **Convolutional Neural Network (CNN)**, and a **Flask web application**.
The goal is to classify images into 10 categories and deploy the model as a simple, interactive web interface.

---

# **I. Features**

* **Deep Learning–Based Image Classification** using a custom CNN trained on CIFAR-10
* **Flask Web Deployment** allowing users to upload or link images for prediction
* **Top-3 Class Predictions** displayed with their confidence scores
* **Data Augmentation & Regularization** to prevent overfitting
* **Training Visualization** using accuracy and loss graphs

---

# **II. Architecture**

### **CNN Pipeline Includes:**

* Normalization & One-Hot Encoding
* Image augmentation via **ImageDataGenerator**
* Convolution + MaxPooling layers
* Dropout for regularization
* Fully connected classification head
* **Adam optimizer** with default parameters
* **ReduceLROnPlateau** and **EarlyStopping** callbacks

This improved architecture increased model robustness and prevented overfitting.

### **Training Performance**

* Final Test Accuracy: **91.74%**

Accuracy and loss curves show stable learning and generalization.

---

# **III. Technologies Used**

* **TensorFlow / Keras** — Model training
* **Flask** — Web deployment
* **OpenCV** — Image processing
* **NumPy, Matplotlib, Seaborn** — Data preprocessing & visualization
* **Google Colab** — GPU training environment

---

# **IV. Dataset**

The project uses the **CIFAR-10 dataset**, containing:

* **60,000 images** (32×32 color)
* **10 distinct classes**
* **50,000 training images**
* **10,000 testing images**

Classes include: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Dataset source:
[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

---

# **V. Flask Deployment**

The Flask interface allows users to:

* Upload an **image file**
* Provide an **image URL**
* Receive the **top 3 predictions** with their confidence percentages

Flask renders a clean prediction page showing:

* Uploaded image preview
* Model predictions
* Confidence distribution

This demonstrates end-to-end deployment of a deep learning model.

---

# **VI. How to Run**

### **1. Clone the Repository**

```bash
https://github.com/ODEDELE2020/CIFAR-10-CNN-and-Flask-Deployment.git
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

> **Note:** Training dependencies (like TensorFlow) are handled automatically in Google Colab.

### **3. Run the Flask App**

```bash
python app.py
```

### **Optional:**

Open the `.ipynb` file to retrain or modify the CNN model.
