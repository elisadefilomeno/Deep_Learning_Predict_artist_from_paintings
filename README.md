# Deep Learning Project: Artist Recognition from Paintings

## Project Overview
This project focuses on using **Deep Learning** techniques to recognize the artist of a painting based on its visual features. The goal is to develop and compare different **Convolutional Neural Network (CNN)** models, both from scratch and using pre-trained architectures, to solve the problem of artist recognition. The project involves dataset preprocessing, model training, evaluation, and visualization techniques to understand how the models make predictions.

## Technologies Used
- **Programming Language**: Python
- **Libraries/Frameworks**: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn
- **Pre-trained Models**: VGG16, ResNet50V2, Xception, InceptionV3
- **Data Augmentation**: RandomFlip, RandomRotation, RandomZoom
- **Visualization**: Intermediate activations, Class activation heatmaps
- **Ensemble Techniques**: Mean averaging, Majority vote

## Dataset
The dataset used in this project is sourced from **Kaggle** and contains paintings from **50 different artists**. After preprocessing, the dataset was reduced to **18 artists** with more than 150 paintings each to ensure sufficient training samples. The dataset is split into:
- **Training/Validation Set**: 80% of the data (4452 paintings)
- **Test Set**: 20% of the data (1112 paintings)

The dataset is balanced using **Data Augmentation** techniques to address class imbalance.

## Key Features
1. **Dataset Preprocessing**:
   - Correction of class names and removal of duplicates.
   - Dataset reduction to focus on artists with more than 150 paintings.
   - Data augmentation to balance the dataset and improve generalization.

2. **CNN from Scratch**:
   - Developed multiple CNN architectures from scratch.
   - Applied techniques like **Dropout**, **L1/L2 regularization**, and **Keras callbacks** (Early Stopping, Model Checkpoint, ReduceLROnPlateau) to combat overfitting.
   - Achieved a **validation accuracy of 69%** and a **test accuracy of 59%** with the best model.

3. **Pre-trained CNN Models**:
   - Utilized pre-trained models like **VGG16**, **ResNet50V2**, **Xception**, and **InceptionV3** for feature extraction and fine-tuning.
   - Fine-tuning improved performance, with **ResNet50V2** achieving the best results: **92% validation accuracy** and **72% test accuracy**.

4. **Visualization**:
   - Visualized intermediate activations to understand how the CNN processes images.
   - Generated heatmaps to identify which parts of the image contributed most to the model's predictions.

5. **Ensemble Learning**:
   - Combined predictions from multiple models using **Mean Averaging** and **Majority Vote** techniques.
   - The ensemble model achieved an overall **test accuracy of 74%**, outperforming individual models.

## Results
- **Best Model**: ResNet50V2 with fine-tuning achieved **92% validation accuracy** and **72% test accuracy**.
- **Ensemble Model**: Mean averaging achieved **74% test accuracy**, the highest among all models.
- **Visualization**: Heatmaps and intermediate activations provided insights into the model's decision-making process.
