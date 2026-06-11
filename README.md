# ♟️ Chess Piece Detection using YOLOv3

> Deep Learning project for chess piece detection using a custom implementation of **YOLOv3 in PyTorch**.

---

## 📚 Table of Contents

- [English Version](#english-version)
  - [Project Overview](#project-overview)
  - [Project Goals](#project-goals)
  - [Dataset](#dataset)
  - [Model Architecture](#model-architecture)
  - [Project Structure](#project-structure)
  - [Technologies Used](#technologies-used)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Results](#results)
  - [Future Improvements](#future-improvements)
- [Russian Version 🇷🇺](#russian-version-)
  - [Описание проекта](#описание-проекта)
  - [Цели проекта](#цели-проекта)
  - [Датасет](#датасет)
  - [Архитектура модели](#архитектура-модели)
  - [Структура проекта](#структура-проекта)
  - [Технологии](#технологии)
  - [Установка](#установка)
  - [Запуск](#запуск)
  - [Результаты](#результаты)
  - [Планы по улучшению](#планы-по-улучшению)

---

# English Version

## Project Overview

This project focuses on **chess piece detection** using a custom implementation of **YOLOv3 (You Only Look Once)** built with **PyTorch**.

The goal of this project is to explore object detection principles and build a deep learning pipeline for recognizing chess pieces on chessboard images.

Unlike projects that rely on ready-made object detection frameworks, this implementation focuses on understanding the internal mechanics of YOLOv3, including:

- custom model architecture
- anchor boxes
- bounding box regression
- YOLO loss function
- dataset processing
- training pipeline
- validation workflow

---

## Project Goals

- Learn object detection fundamentals
- Implement a custom YOLOv3 architecture
- Work with labeled image datasets
- Train an object detection model
- Understand anchor boxes and bounding boxes
- Gain practical experience in PyTorch Computer Vision

---

## Dataset

The dataset contains labeled chessboard images with annotated chess pieces.

### Dataset Split

| Split | Images |
|--------|---------|
| Train | 606 |
| Validation | 58 |

### Classes

The model detects **12 chess piece classes**:

- White King
- White Queen
- White Rook
- White Bishop
- White Knight
- White Pawn
- Black King
- Black Queen
- Black Rook
- Black Bishop
- Black Knight
- Black Pawn

---

## Model Architecture

This project uses a **custom implementation of YOLOv3** in **PyTorch**.

Implemented components include:

- Darknet-inspired backbone
- Detection heads
- Anchor boxes
- Bounding box regression
- YOLO loss function
- Custom dataset loader
- Training loop

The implementation was created for educational purposes and practical deep learning experience.

---

## Project Structure

```text
chess_detectioin_YOLOv3/
│── notebooks/
│   └── edu.ipynb
│
│── main.py
│── requirements.txt
│── README.md
```

### File Description

| File | Description |
|------|-------------|
| `notebooks/edu.ipynb` | Main notebook with YOLOv3 implementation and training |
| `main.py` | Auxiliary Python file |
| `requirements.txt` | Project dependencies |
| `README.md` | Project documentation |

---

## Technologies Used

### Programming Language
- Python

### Deep Learning
- PyTorch

### Computer Vision
- OpenCV
- PIL

### Data Processing
- NumPy
- Pandas

### Visualization
- Matplotlib

### Environment
- Jupyter Notebook

---

## Installation

Clone the repository:

```bash
git clone https://github.com/himynameisartem/chess_detectioin_YOLOv3.git
cd chess_detectioin_YOLOv3
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/edu.ipynb
```

The notebook contains:

- dataset preparation
- model implementation
- training pipeline
- validation
- prediction experiments

---

## Results

This project demonstrates a complete deep learning workflow for object detection:

**data → training → optimization → validation → predictions**

During training, model loss decreases over epochs, providing practical experience with:

- object detection
- bounding boxes
- anchor boxes
- CNN architectures
- computer vision pipelines

---

## Future Improvements

Planned improvements:

- Add `predict.py` for inference
- Add prediction visualization
- Add evaluation metrics (`mAP`)
- Improve project structure
- Export trained weights
- Create deployment API

---

# Russian Version 🇷🇺

## Описание проекта

Проект посвящён **детекции шахматных фигур** с использованием собственной реализации **YOLOv3 на PyTorch**.

Основная цель проекта — изучить принципы **object detection** и построить полноценный deep learning pipeline для распознавания фигур на изображениях шахматной доски.

В отличие от проектов на готовых библиотеках, здесь реализованы ключевые компоненты YOLOv3 вручную:

- архитектура модели
- anchor boxes
- bounding box regression
- функция потерь
- обработка датасета
- цикл обучения
- валидация

---

## Цели проекта

- Изучить object detection
- Реализовать собственную YOLOv3
- Научиться работать с размеченными изображениями
- Построить pipeline обучения модели
- Изучить bounding boxes и anchor boxes
- Получить практический опыт в PyTorch CV

---

## Датасет

Датасет содержит изображения шахматных досок с размеченными фигурами.

### Разделение данных

| Split | Images |
|--------|---------|
| Train | 606 |
| Validation | 58 |

### Классы

Модель обучается распознавать **12 классов шахматных фигур**:

- Белый король
- Белый ферзь
- Белая ладья
- Белый слон
- Белый конь
- Белая пешка
- Чёрный король
- Чёрный ферзь
- Чёрная ладья
- Чёрный слон
- Чёрный конь
- Чёрная пешка

---

## Архитектура модели

В проекте используется **собственная реализация YOLOv3**.

Реализованы:

- backbone в стиле Darknet
- detection heads
- anchor boxes
- bounding box regression
- YOLO loss
- загрузчик данных
- цикл обучения

---

## Структура проекта

```text
chess_detectioin_YOLOv3/
│── notebooks/
│   └── edu.ipynb
│
│── main.py
│── requirements.txt
│── README.md
```

---

## Технологии

### Язык программирования
- Python

### Deep Learning
- PyTorch

### Computer Vision
- OpenCV
- PIL

### Обработка данных
- NumPy
- Pandas

### Визуализация
- Matplotlib

### Среда разработки
- Jupyter Notebook

---

## Установка

Клонировать репозиторий:

```bash
git clone https://github.com/himynameisartem/chess_detectioin_YOLOv3.git
cd chess_detectioin_YOLOv3
```

Установить зависимости:

```bash
pip install -r requirements.txt
```

---

## Запуск

Запустить Jupyter Notebook:

```bash
jupyter notebook
```

Открыть:

```text
notebooks/edu.ipynb
```

Ноутбук содержит:

- подготовку данных
- реализацию модели
- обучение
- валидацию
- эксперименты с предсказаниями

---

## Результаты

Проект демонстрирует полный workflow deep learning модели:

**данные → обучение → оптимизация → валидация → предсказания**

Во время обучения наблюдается снижение `loss`, что позволяет получить практический опыт работы с:

- object detection
- bounding boxes
- anchor boxes
- CNN архитектурами
- computer vision pipeline

---

## Планы по улучшению

- Добавить `predict.py`
- Добавить визуализацию detections
- Добавить метрику `mAP`
- Улучшить структуру проекта
- Добавить сохранение лучших весов модели
- Сделать API для inference

---

⭐ **If you like this project, feel free to star the repository!**
