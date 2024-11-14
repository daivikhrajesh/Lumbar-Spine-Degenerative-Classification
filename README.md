#**Lumbar Spine Degenerative Classification** 🏥

## 📢 **Project Overview**

Low back pain is a prevalent health issue worldwide, affecting millions of people and often resulting in disability. It is frequently caused by degenerative spine conditions, such as the narrowing of the spinal canal, subarticular recesses, or neural foramen, which can compress nerves and lead to pain. Radiologists use Magnetic Resonance Imaging (MRI) scans to assess and diagnose these conditions, which is crucial for deciding the proper course of treatment, including potential surgery.

In collaboration with the **Radiological Society of North America (RSNA)** and the **American Society of Neuroradiology (ASNR)**, this project aims to explore the potential of artificial intelligence (AI) in detecting and classifying degenerative spine conditions using lumbar spine MRI scans.

This project demonstrates the power of ensemble learning techniques in medical image classification. By using multiple models, this system classifies medical images into four categories based on severity. The final output is achieved by combining the predictions of different deep learning models to improve accuracy and generalization.

This project involves building an AI model capable of classifying five degenerative conditions of the lumbar spine:

1. Left Neural Foraminal Narrowing
2. Right Neural Foraminal Narrowing
3. Left Subarticular Stenosis
4. Right Subarticular Stenosis
5. Spinal Canal Stenosis

The MRI scans are classified across five intervertebral disc levels:

- L1/L2
- L2/L3
- L3/L4
- L4/L5
- L5/S1

Each condition is graded based on severity: **Normal/Mild**, **Moderate**, or **Severe**.

## Project Files and Dataset

The dataset provided for this competition includes several important files that contain the imaging data and corresponding labels:

### Files

- **`train.csv`**: Contains the study ID, target labels (severity for each condition and disc level), and partial or incomplete labels for some studies.
- **`train_label_coordinates.csv`**: Coordinates (x, y) of labeled conditions in the MRI scans, along with study IDs, series IDs, and instance numbers.
- **`sample_submission.csv`**: A template for making submissions, including the `row_id` (combining study ID, condition, and disc level) and the columns for predicted severity (`normal_mild`, `moderate`, and `severe`).
- **`train_images/`**: The DICOM (.dcm) MRI scan files for the training set, organized by `study_id` and `series_id`.
- **`train_series_descriptions.csv`**: Contains descriptions of MRI scan series and their orientations.

### Conditions & Levels

The goal is to predict the severity for each condition at each disc level, where the severity levels are:

- **Normal/Mild**
- **Moderate**
- **Severe**

### Dataset Size

- Total size: ~35.34 GB
- The dataset contains DICOM files for MRI scans, which need to be processed and visualized.

## 💻 **Technologies Used**
- **Deep Learning Frameworks:** 
  - PyTorch 🤖
- **Libraries:** 
  - `torchvision` 🖼️
  - `Pandas` 📊
  - `NumPy` 🔢
  - `Matplotlib` 📈
- **Image Processing:**
  - DICOM Image Loading & Transformation
  - Data Augmentation and Preprocessing
- **Model Optimization:**
  - Hyperparameter Tuning 🔧
  - Cross-Validation 🔍

## Project Components

### Data Preprocessing and Visualization

1. **DICOM Image Loading**: The DICOM files are loaded using the `pydicom` library, which allows access to pixel arrays and metadata.
2. **Image Visualization**: Using libraries like `matplotlib`, `plotly`, and `seaborn`, DICOM images are visualized. Coordinate points corresponding to labeled conditions are overlaid on the images for better understanding.
3. **3D Visualization**: The MRI scans (slices) are stacked and visualized in 3D using the `marching_cubes` algorithm and `Poly3DCollection` for a better grasp of anatomical structures.

### Model Training

The training process uses data from `train.csv`, which contains severity labels for the five conditions across five spinal levels. A **Random Forest Classifier** is trained to predict these labels. Here's a summary of the steps involved:

1. **Data Transformation**: Data is preprocessed and melted to separate conditions and levels, and severity labels are encoded into numerical values using `LabelEncoder`.
2. **Feature Engineering**: Features like `study_id`, `condition`, and `level` are used, and one-hot encoding is applied for categorical features.
3. **Model Fitting**: The `RandomForestClassifier` is trained on the transformed features and severity labels.
4. **Prediction**: The model makes predictions for the test set, with probabilities for each severity level (Normal/Mild, Moderate, Severe).

### Submission

The submission file follows the structure of `sample_submission.csv`. Predictions for each condition at each level are stored in the columns `normal_mild`, `moderate`, and `severe`.

### Visualization of Results

A line plot shows the predicted values for Normal/Mild, Moderate, and Severe across different conditions and levels in the test set.

## Getting Started

### 🛠️ Prerequisites

The following Python libraries are required to run this project:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly`
- `pydicom`
- `cv2` (OpenCV)
- `skimage`
- `scikit-learn`
- `plotly.graph_objects`
- `mpl_toolkits`

### 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/daivikhrajesh/rsna-lumbar-spine-classification.git
   cd rsna-lumbar-spine-classification
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from the RSNA page and place the files in the `data/` directory.

4. Run the notebooks in `notebooks/` to explore the data, preprocess it, and train the model.

## 📜 **License**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## **Conclusion**

This project demonstrates the use of artificial intelligence in the detection and classification of lumbar spine degenerative conditions. By processing MRI scans, visualizing data, and building a predictive model, it aims to assist radiologists in diagnosing spine conditions more efficiently.

