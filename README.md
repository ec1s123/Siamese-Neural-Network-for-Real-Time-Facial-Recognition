# 🧠 Face Verification with Siamese Neural Networks

A real-time facial verification system using a custom-trained Siamese Neural Network in TensorFlow. Inspired by the paper ["Siamese Neural Networks for One-shot Image Recognition"](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf), this project compares two face images and predicts whether they belong to the same identity using high-dimensional embeddings and distance-based similarity.

![Demo Screenshot](./demo.png) <!-- optional if you have a screenshot -->

---

## 🚀 Features

- **One-shot Face Verification** using a 39M-parameter Siamese CNN
- **4096-dimensional embeddings** per image for deep feature representation
- **Real-time verification** with OpenCV webcam input
- **Contrastive training** on anchor-positive-negative pairs
- **Mixed Precision** and `prefetch` pipeline for optimized training
- **Custom dataset** blended with benchmark LFW images

---

## 🧠 Architecture

- Shared Convolutional Encoder → Dense(4096) embedding
- L1 Distance layer compares embeddings
- Dense(1, sigmoid) outputs match probability


---

## 📦 Dataset

- **Anchor/Positive**: Captured via webcam (`OpenCV`)
- **Negative**: Sampled from [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
- Dataset is loaded using TensorFlow's `tf.data` pipeline with `.cache()`, `.shuffle()`, and `.prefetch()` for performance

---

## 🧪 Training

```bash
python train.py


🛠️ Technologies
Python

TensorFlow

OpenCV

NumPy

📄 Citation
Inspired by:
Koch, Gregory, Richard Zemel, and Ruslan Salakhutdinov. "Siamese neural networks for one-shot image recognition." ICML Deep Learning Workshop. 2015.



