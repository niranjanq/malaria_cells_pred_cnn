# 🦠 Malaria Cell Classification using CNN

This project builds a Convolutional Neural Network (CNN) model to classify red blood cell images as **Parasitized** or **Uninfected** using a publicly available dataset. It also supports making predictions on new unseen images.

---

## 📂 Dataset

- **Source**: [Kaggle – Cell Images for Malaria Detection](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- The dataset contains images of parasitized and uninfected cells under a microscope, organized in separate folders.

---

## 🚀 Features

- Preprocesses and loads all malaria cell images
- Builds a custom CNN model using TensorFlow and Keras
- Trains the model and visualizes accuracy/loss trends
- Evaluates the model using validation accuracy and prediction visualization
- Makes predictions on custom images not seen during training
- Saves the trained model for future reuse

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, PIL
- Google Colab / Jupyter Notebook

---

## 🧠 Model Architecture

- Multiple `Conv2D` layers with `ReLU` activation
- `MaxPooling2D` for downsampling
- `Dropout` for regularization
- `Flatten` and `Dense` layers for classification
- Final layer: `sigmoid` for binary classification

---

## 📊 Training Insights

- Achieved validation accuracy of **95%+**
- Detected overfitting and addressed it with dropout and early stopping
- Accuracy and loss curves included in the notebook

---

## 📷 Prediction on Custom Images

You can upload a malaria cell image (outside the dataset) and the model will predict:
- **Parasitized**
- **Uninfected**

Sample code:
```python
# Load and preprocess a custom image
img = Image.open("test_cell.jpg").resize((64, 64)).convert('RGB')
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

# Predict
prediction = model.predict(img_array)
print("Predicted:", "Uninfected" if prediction[0][0] > 0.5 else "Parasitized")
