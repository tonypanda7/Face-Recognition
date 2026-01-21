Overview

This project implements 1-vs-Many face verification using Siamese Neural Networks.
1-vs-Many: One input face is compared against multiple reference images of a single authorized person
Not Many-vs-Many: The system does not identify or classify among multiple identities

This repository contains two Siamese Neural Network–based face verification implementations:
  1.Image upload–based verification using L2 distance + Contrastive Loss
  2.Real-time face verification using L1 distance + Binary Cross-Entropy, deployed via a Kivy app

DATASET sturucture
data/
│
├── anchor/
│   ├── a1.jpg
│   ├── a2.jpg
│   └── ...
│
├── positive/
│   ├── p1.jpg
│   ├── p2.jpg
│   └── ...
│
└── negative/
    ├── n1.jpg
    ├── n2.jpg
    └── ...


code to run the face-recognition-app
py faceid.py
