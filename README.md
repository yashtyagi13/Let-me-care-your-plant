# Plant Disease Detection Using Classical Machine Learning Algorithms and Image Processing

## Table of Contents

- [Introduction](#introduction)
- [About the Dataset](#about-the-dataset)
- [Properties of Images](#properties-of-images)
- [Steps Involved](#steps-involved)
  - [Data Preprocessing](#data-preprocessing)
  - [Modeling](#modeling)
  - [Prediction](#prediction)
- [Conclusion](#conclusion)
- [Directory Structure](#directory-structure)

## Introduction

Plant disease detection is a critical issue in the application of technology to agriculture. While extensive research has been done using deep learning and neural networks for identifying whether a plant is healthy or diseased, new techniques are continually being developed. This project presents an approach for detecting whether a plant leaf is healthy or unhealthy using classical machine learning algorithms, with data preprocessing carried out through image processing techniques.

## About the Dataset

The dataset for this project is sourced from the PlantVillage Dataset, which is available [here](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color). Specifically, the data used is from the "color" folder within the "raw" directory of the repository. This project focuses on images of apple leaves, categorized into two folders: Diseased and Healthy. The Diseased folder includes leaves affected by Apple Scab, Black Rot, or Cedar Apple Rust, while the Healthy folder contains images of green, healthy leaves.

## Properties of Images

- **File Type:** JPG
- **Dimensions:** 256 x 256 pixels
- **Horizontal and Vertical Resolution:** 96 dpi
- **Bit Depth:** 24

## Steps Involved

### Data Preprocessing

1. **Load Original Images:** A total of 800 images for each class (Diseased and Healthy) are loaded.
2. **Convert RGB to BGR:** Since OpenCV (a Python library for image processing) uses BGR format, the images are converted accordingly.
3. **Convert BGR to HSV:** HSV separates the image intensity (luma) from color information (chroma), which is useful for various applications such as histogram equalization and robustness to lighting changes.
4. **Image Segmentation:** To isolate the leaf from the background, color extraction and segmentation are performed.
5. **Global Feature Descriptor:** Features are extracted using three descriptors:
   - **Color:** Color Channel Statistics (Mean, Standard Deviation) and Color Histogram.
   - **Shape:** Hu Moments, Zernike Moments.
   - **Texture:** Haralick Texture, Local Binary Patterns (LBP).
6. **Feature Stacking:** The extracted features are combined using the numpy `np.stack` function.
7. **Label Encoding:** The labels of the images are encoded numerically for better machine understanding.
8. **Train-Test Split:** The dataset is divided into training (80%) and testing (20%) sets.
9. **Feature Scaling:** Min-Max Scaler is used to scale the features between 0 and 1, ensuring consistent feature magnitudes.
10. **Save Features:** The features are saved in an HDF5 file, which supports large, complex data structures.

### Modeling

The model is trained using the following seven machine learning algorithms:
- Logistic Regression
- Linear Discriminant Analysis
- K-Nearest Neighbors
- Decision Trees
- Random Forest
- Naïve Bayes
- Support Vector Machine

The models are validated using 10-fold cross-validation to ensure robustness.

### Prediction

The best-performing model, the Random Forest Classifier, is trained on the entire dataset. The accuracy for the testing set is then predicted using the `predict` function, achieving an accuracy of 97%.

## Conclusion

This project demonstrates an effective method for plant disease detection using classical machine learning algorithms combined with image processing techniques. By focusing on feature extraction and proper data preprocessing, we achieve high accuracy without relying on deep learning models.

## Directory Structure

- **Utils:** Contains Python scripts for label conversion of images in the training folders.
- **Image Classification:** Contains the training dataset and the Jupyter Notebook for plant disease detection.
- **Testing Notebook:** Provides detailed specifications of the functions applied to the leaf images.
