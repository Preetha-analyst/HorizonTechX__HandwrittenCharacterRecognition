# Handwritten Character Recognition using CNN

A Deep Learning-based Handwritten Character Recognition system developed using **Python**, **TensorFlow**, **Keras**, and **OpenCV**. The model is trained on the **EMNIST ByClass** dataset to recognize handwritten digits, uppercase letters, and lowercase letters.

---

## Features

- Handwritten character recognition
- EMNIST ByClass dataset (62 classes)
- CNN-based deep learning model
- Image preprocessing using OpenCV
- Character segmentation
- Automatic character prediction
- Bounding box visualization
- Output image generation

---

## Technologies Used

- Python 3.11
- TensorFlow
- Keras
- OpenCV
- NumPy
- SciPy
- EMNIST Dataset

---

## Project Structure

```
Handwritten-Character-Recognition/
│
├── train_model.py
├── ocr_predict.py
├── advanced_emnist_model.h5
├── requirements.txt
├── README.md
├── sample_images/
├── outputs/
└── report/
```

---

## Dataset

- **Dataset:** EMNIST ByClass
- **Classes:** 62
- Digits (0–9)
- Uppercase Letters (A–Z)
- Lowercase Letters (a–z)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Handwritten-Character-Recognition.git
```

Go to the project folder:

```bash
cd Handwritten-Character-Recognition
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train_model.py
```

---

## Run OCR

```bash
python ocr_predict.py
```

---

## Output

The system:

- Reads the input image
- Detects handwritten characters
- Predicts each character
- Displays recognized text
- Saves the output image with bounding boxes

---

## Future Enhancements

- Word-level recognition
- Sentence recognition
- Real-time camera OCR
- Mobile application
- Web application

---

## Author

**Preetha S**

---

## License

This project is developed for learning and internship purposes. 

# HorizonTechX__HandwrittenCharacterRecognition
