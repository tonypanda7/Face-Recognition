<p align="center">
    <h1 align="center">FACE RECOGNITION</h1>
</p>
<p align="center">
    <em><code>❯ Tonypanda7 </code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/tonypanda7/Face-Recognition?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/tonypanda7/Face-Recognition?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/tonypanda7/Face-Recognition?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/tonypanda7/Face-Recognition?style=flat&color=0080ff" alt="repo-language-count">
</p>
<p align="center">
		<em>Built with the tools and technologies:</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
</p>

<br>

## Overview

This project implements 1-vs-Many face verification using Siamese Neural Networks.
1-vs-Many: One input face is compared against multiple reference images of a single authorized person
Not Many-vs-Many: The system does not identify or classify among multiple identities

This repository contains two Siamese Neural Network–based face verification implementations:
  1.Image upload–based verification using L2 distance + Contrastive Loss
  2.Real-time face verification using L1 distance + Binary Cross-Entropy, deployed via a Kivy app

```
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
```

## Installation Run 

```bash
python -m venv .venv
source .venv/bin/activate
py faceid.py
```
