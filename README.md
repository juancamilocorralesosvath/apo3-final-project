
# Pose-Based Human Activity Recognition (HAR)
This project is a part of the  **Algoritmos y programacion 3** course in the Bachelor's Degree in Software Engineering, Universidad Icesi, Cali Colombia. 

#### -- Project Status: Active

## Contributing Members

|Name     |  Github   | 
|---------|-----------------|
|Luis Manuel Rojas Correa| https://github.com/Lrojas898        |
|William Joseph Verdesoto |     https://github.com/BillyJoe121    |
|Juan Camilo Corrales Osvath |     https://github.com/juancamilocorralesosvath   |


**Instructor: Milton Orlando Sarria Paja https://github.com/miltonsarria**


## Project Intro/Objective
The primary objective of this project is to develop a machine learning system capable of classifying specific human activities in real-time from video streams. By leveraging pose estimation to track key body landmarks, the system analyzes postural and movement data to distinguish between actions such as walking, sitting, turning, and squatting. The project aims to create a robust classification pipeline, from data preprocessing and feature engineering to model training and evaluation. This work serves as a foundation for applications in areas like automated surveillance, fitness tracking, or ergonomic analysis.

### Methods Used
*   Machine Learning
*   Predictive Modeling
*   Data Visualization
*   Feature Engineering
*   Time-Series Analysis
*   Signal Filtering (Savitzky-Golay)
*   Dimensionality Reduction (PCA, t-SNE)
*   Class Imbalance Handling (SMOTE)
*   Descriptive & Inferential Statistics

### Technologies
*   Python
*   Google Colab, Jupyter Notebook
*   **Data Analysis & Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Data Visualization:** Plotly, Matplotlib, Seaborn
*   **Pose Estimation:** MediaPipe

## Project Description
This project addresses the challenge of Human Activity Recognition (HAR) using computer vision, following the guidelines of the Artificial Intelligence I course at Universidad ICESI. The primary data source consists of video clips of individuals performing a predefined set of activities (e.g., `approach`, `squats`, `turn`). These videos were manually annotated to create labeled segments for supervised learning.

The core of the methodology involves a multi-stage data processing pipeline:
1.  **Pose Estimation:** Google's `MediaPipe` library was used to extract the 3D coordinates of key body landmarks (shoulders, hips, knees, ankles, etc.) from each video frame.
2.  **Data Cleaning & Normalization:** To ensure the model is invariant to the subject's position and distance from the camera, the raw landmark data was normalized relative to the hip center. A Savitzky-Golay filter was then applied to the time-series data to smooth the signals and reduce detection noise.
3.  **Feature Engineering:** A comprehensive set of biomechanical features was engineered to capture both static posture and dynamic movement. These included key joint angles (e.g., knee and hip flexion), trunk inclination, and the velocity of individual landmarks.
4.  **Modeling & Evaluation:** The central hypothesis was that these engineered features could effectively train classifiers to distinguish between activities. We explored and evaluated several models, including Random Forest, SVM, and XGBoost. A significant challenge emerged during the initial modeling phase: an approach based on static, aggregated features (mean/std per segment) yielded poor performance (~50% accuracy) due to a massive loss of temporal information.
5.  **Pivoting and Optimization:** To overcome this, the project pivoted to a more robust temporal approach, segmenting the data into windows of 30 frames. This not only preserved the sequential nature of the movements but also significantly augmented the training data, boosting model accuracy to ~69%. Further optimization was performed on the best-performing model (XGBoost) using the SMOTE technique to address the inherent class imbalance in the dataset, resulting in a more equitable and robust final model.

## Getting Started

*   Clone this repo.

## Submission 2: Exploratory Data Analysis (EDA)

The work developed for this submission is contained within the following Jupyter Notebook:

`Entrega2/notebooks/EDA_Proyecto_Final.ipynb`

The necessary data for running and reproducing the analysis can be found in the folder:

`Entrega2/data/`



