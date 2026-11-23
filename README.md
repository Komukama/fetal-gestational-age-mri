
# Fetal Gestational Age Prediction from Fetal Brain MRI

## Overview

This project investigates the application of machine learning and deep learning techniques for predicting fetal gestational age (GA) from fetal brain Magnetic Resonance Imaging (MRI).

The study uses three orthogonal 2D MRI planes:

- Axial
- Coronal
- Sagittal

The images are combined to represent fetal brain anatomy from multiple perspectives. Classical machine learning models and convolutional neural networks (CNNs) are evaluated to investigate their performance in estimating gestational age in days.

The project also explores multi-plane attention, model fine-tuning, prediction errors, and visual interpretability.

---

## Objectives

The main objectives of this project are:

1. Develop a machine learning pipeline for predicting fetal gestational age from fetal brain MRI.
2. Compare classical regression models with deep learning models.
3. Investigate the contribution of axial, coronal, and sagittal MRI planes through a multi-plane attention architecture.
4. Evaluate model performance using independent training, validation, and test partitions.
5. Analyze prediction errors across different gestational-age ranges.
6. Investigate model behavior using Grad-CAM and attention-weight analysis.

---

## Dataset

The study uses the fetal brain MRI dataset from Stanford Lucile Packard Children's Hospital.

**Dataset:** Fetal Brain MRI from Stanford Lucile Packard Children's Hospital

**Stanford Digital Repository:**  
https://purl.stanford.edu/sf714wg0636

The dataset contains fetal brain MRI examinations and corresponding gestational-age labels.

The dataset includes image information from three anatomical planes:

- Axial
- Coronal
- Sagittal

The experiment uses 2D middle-slice images from each plane rather than complete 3D MRI volumes.

### Dataset preprocessing

The preprocessing pipeline includes:

- Matching image filenames with the dataset labels.
- Validating patient identifiers and gestational-age labels.
- Checking the availability of all three MRI planes.
- Filtering incomplete records.
- Applying the gestational-age range filter.
- Resizing images.
- Normalizing image values.
- Constructing three-plane input representations.

After preprocessing, 708 examinations from 708 unique patients were retained.

---

## Data Partitioning

The dataset was partitioned at patient level to reduce the risk of data leakage between subsets.

| Dataset | Number of examinations |
|---|---:|
| Training | 495 |
| Validation | 106 |
| Testing | 107 |
| Total | 708 |

The split was approximately 70% training, 15% validation, and 15% testing.

Random seed: `42`

The validation set was used for model selection and fine-tuning. The held-out test set was used for final evaluation.

---

## Models Evaluated

Five regression models were investigated.

### Classical Machine Learning Models

1. Linear Regression
2. Random Forest Regression
3. Support Vector Regression (RBF kernel)

Classical models use handcrafted image features derived from the MRI planes, including Histograms of Oriented Gradients (HOG) and intensity histograms.

### Deep Learning Models

4. ResNet-18
5. Multi-Plane Attention ResNet-50

The multi-plane attention model uses a shared ResNet-50 feature encoder to extract features independently from the axial, coronal, and sagittal images.

A learned attention mechanism assigns weights to the plane-specific feature representations. The weighted features are combined and passed to a regression head to predict gestational age.

The attention model is a practical reconstruction inspired by published multi-plane fetal brain MRI research and is not an exact reproduction of the original published architecture.

---

## Model Training

The deep learning models were trained using PyTorch.

The training process includes:

- Image preprocessing and normalization.
- Transfer learning using pretrained ResNet weights.
- AdamW optimization.
- Smooth L1 loss.
- Learning-rate scheduling.
- Gradient clipping.
- Validation-based model selection.
- Early stopping.

The selected multi-plane attention ResNet-50 was subsequently fine-tuned by freezing its encoder and updating its attention module and regression head.

---

## Evaluation Metrics

Model performance is evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Lin's Concordance Correlation Coefficient (CCC)
- MAE in weeks

MAE and RMSE are reported in days.

---

## Results

### Validation Model Comparison

The multi-plane attention ResNet-50 achieved the lowest validation MAE among the five evaluated models.

| Model | Validation MAE (days) |
|---|---:|
| Multi-Plane Attention ResNet-50 | 10.256 |
| Linear Regression | 19.520 |
| Random Forest | 25.475 |
| SVR (RBF) | 29.846 |
| ResNet-18 | 137.323 |

The multi-plane attention ResNet-50 was selected based on validation MAE.

### Final Test Results

The fine-tuned multi-plane attention ResNet-50 achieved the following performance on the held-out test set:

| Metric | Test Result |
|---|---:|
| MAE | 9.304 days |
| RMSE | 12.963 days |
| R² | 0.8760 |
| Lin's CCC | 0.9367 |
| MAE | 1.329 weeks |

These results represent the performance obtained on the held-out test set from the dataset used in this experiment. They do not establish clinical readiness or external generalizability.

---

## Fine-Tuning

The selected multi-plane attention ResNet-50 was fine-tuned by:

- Freezing the ResNet-50 encoder.
- Training the attention module.
- Training the regression head.
- Using AdamW with a reduced learning rate.
- Using Smooth L1 loss.
- Monitoring validation MAE.
- Saving the best fine-tuned checkpoint.

The fine-tuning process improved the test MAE from 9.748 days to 9.304 days.

---

## Model Interpretability

### Grad-CAM

Grad-CAM was applied to the final convolutional layer to visualize image regions associated with model predictions.

Separate visualizations were generated for:

- Axial MRI
- Coronal MRI
- Sagittal MRI

These visualizations provide qualitative insights into the model's spatial responses but do not establish causal explanations or clinical anatomical importance.

### Attention Analysis

Attention weights were extracted from the multi-plane attention model across the test set.

Mean attention weights:

| MRI Plane | Mean Attention |
|---|---:|
| Axial | 0.3568 |
| Coronal | 0.3273 |
| Sagittal | 0.3159 |

The weights describe the relative contribution of each plane's feature representation within the model. They should not be interpreted as direct measures of clinical importance.

---

## Gestational-Age-Band Analysis

Test-set prediction errors were analyzed across five gestational-age bands:

- Below 24 weeks
- 24–28 weeks
- 28–32 weeks
- 32–36 weeks
- 36 weeks and above

Mean Absolute Error was calculated for each band.

Bootstrap confidence intervals were estimated using 10,000 resamples to assess uncertainty in the subgroup MAE estimates.

This analysis examines error variation across gestational-age ranges and does not constitute demographic fairness evaluation.

---

## Project Structure

A repository structure is:

```text
fetal-gestational-age-mri/
│
├── fetal_gestational_age_prediction.ipynb
│  
│
├── results_fetal_ga/
│   ├── figures/
│   └── results/
│
│
├── README.md
├── requirements.txt
└── .gitignore
```

The exact directory structure may be adjusted to match the notebook and scripts used in the experiment.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Komukama/fetal-gestational-age-mri.git
```

Navigate into the project directory:

```bash
cd fetal-gestational-age-mri
```

Create a virtual environment:

```bash
python -m venv venv
```

### Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

Linux / WSL:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Alternatively, open the notebook using JupyterLab or Google Colab.

---

## Reproducibility

The experimental workflow uses:

- Random seed: 42
- Patient-level dataset partitioning
- Training, validation, and held-out test sets
- Validation-based model selection
- Separate fine-tuning of the selected model
- Final evaluation using the held-out test set

For reproducibility, users should record:

- Python version
- Package versions
- Hardware configuration
- GPU model
- Random seed
- Dataset version
- Model configuration
- Training parameters

The dataset should be obtained from its original source and used according to its access, licensing, and redistribution conditions.

---

## Limitations

The study has several limitations:

1. The experiment uses a single-source dataset.
2. Independent external validation was not performed.
3. The model uses three 2D middle slices rather than full 3D MRI volumes.
4. Classical models and CNNs use different feature representations.
5. The attention model is a reconstruction inspired by published work rather than an exact reproduction.
6. Grad-CAM analysis is qualitative.
7. Some gestational-age subgroups have limited sample sizes.
8. The findings do not establish clinical readiness.

Further research should investigate external validation, broader patient populations, different MRI acquisition protocols, additional model architectures, and prospective clinical evaluation.

---

## References

1. Lee, L. H., et al. (2023). Machine learning for accurate estimation of fetal gestational age based on ultrasound images. *npj Digital Medicine*.

2. Benson, M., et al. (2025). Fetal gestational age estimation using artificial intelligence on non-targeted ultrasound images and video. *npj Digital Medicine*.

3. Shen, L., et al. (2022). Attention-guided deep learning for gestational age prediction using fetal brain MRI. *Scientific Reports*.

4. Stanford University. Fetal Brain MRI from Stanford Lucile Packard Children's Hospital. Stanford Digital Repository. https://purl.stanford.edu/sf714wg0636

5. Self, A., et al. (2022). Second and third trimester estimation of gestational age using ultrasound or maternal symphysis-fundal height measurements: A systematic review. *BJOG*.

6. Bradburn, E., et al. (2024). Accuracy of prenatal and postnatal biomarkers for estimating gestational age: A systematic review and meta-analysis. *eClinicalMedicine*.

7. Dan, T., et al. (2023). DeepGA for automatically estimating fetal gestational age through ultrasound imaging. *Artificial Intelligence in Medicine*.

8. Lee, C., et al. (2023). Development of a machine learning model for sonographic assessment of gestational age. *JAMA Network Open*.

9. Stringer, J. S. A., et al. (2024). Diagnostic accuracy of an integrated AI tool to estimate gestational age from blind ultrasound sweeps. *JAMA*.

---

## Disclaimer

This project is for research and educational purposes.

The models developed in this study are not intended to replace clinical assessment, professional medical judgment, or established methods of pregnancy dating.

The reported results should not be interpreted as evidence of clinical validity or readiness for deployment.

---

## Author

**Anita Komukama**

Research project: Machine Learning-Based Prediction of Fetal Gestational Age using Fetal Brain Magnetic Resonance Imaging(MRI)








