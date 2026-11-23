# UAV-XAI

This repository contains the source code and link to the benchmark dataset for **UAV-XAI**, a lightweight Random Forest-based UAV network security framework integrating Gini feature importance, SHapley Additive exPlanations (SHAP), and Local Interpretable Model-Agnostic Explanations (LIME) to deliver explainable and dependable anomaly detection.

More details on the methodology and experimental evaluation are available in the accompanying paper, *UAV-XAI*.

---

## 📂 Repository Contents

*   **`UAV-XAI.py`**: The main executable script that implements preprocessing, data partitioning, a lightweight Random Forest classifier, Explainable AI (XAI) models, and an empirical feature ablation study.

---

## 📊 Dataset Generation

The dataset used in the paper is generated from the **ISOT Drone 2024** dataset available at the https://onlineacademiccommunity.uvic.ca/isot/2024/12/05/drone-datasets.

*   **Kaggle Benchmark Dataset ISOT Drone 2024 Link**: https://www.kaggle.com/datasets/shalam3/isot-drone-master-dataset
*   **Kaggle Benchmark Dataset UAV-IDS 2025 Link**: https://www.kaggle.com/datasets/qinglizeng1997/uavids-2025
*   **Important Note**: The name of the dataset files (`isot_drone_master_dataset.csv` and `UAVIDS-2025.csv`) are **hard-coded** directly inside the script. Please ensure these files are downloaded and placed in your working directory prior to running the pipeline.

---

## 🚀 Usage

Execute the main analytical and explanatory pipeline using the following command:

```bash
python UAV-XAI.py [-h] [-stratify] [-temporal] [-isot-2024]

options:
  -h, --help  show this help message and exit
  -stratify   ; Set if you want to stratify the split by class labels. Only set one of them (stratify or temporal)
  -temporal   ; Set if you want to split based on temporal order. Only set one of them (stratify or temporal)
  -isot-2024  ; Set if using the ISOT Drone 2024 dataset, otherwise by default it uses the UAVIDS-2025 dataset
```

---

## 📦 Installation

1. Clone the repository and navigate to the project folder.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 📉 Generated Outputs

Running the pipeline populates your directory with the following metric logs, data subsets, and evaluation visualizations:

### Data & Results Summary
*   `drone_security_results.txt`

### Base Metrics & Comparisons
*   `dataset_distribution.pdf`
*   `random_forest_confusion_matrix.pdf`
*   `per_class_xai_comparison.pdf`

### Feature-Specific Global XAI Comparisons
*   `true_xai_comparison_benign.pdf`
*   `true_xai_comparison_dos.pdf`
*   `true_xai_comparison_injection.pdf`
*   `true_xai_comparison_ip-spoofing.pdf`
*   `true_xai_comparison_manipulation.pdf`
*   `true_xai_comparison_mitm.pdf`
*   `true_xai_comparison_password-cracking.pdf`
*   `true_xai_comparison_replay.pdf`
*   `true_xai_comparison_unauth.pdf`
*   `true_xai_comparison_video.pdf`

### Local LIME Case Studies & Visual Explanations
*   `lime_case_study_benign_correct_classification_id10889.pdf`
*   `lime_case_study_benign_correct_classification_id13919.pdf`
*   `lime_case_study_dos_correct_classification_id19474.pdf`
*   `lime_case_study_dos_correct_classification_id3144.pdf`
*   `lime_case_study_injection_correct_classification_id19260.pdf`
*   `lime_case_study_injection_correct_classification_id6983.pdf`
*   `lime_case_study_ip-spoofing_correct_classification_id16375.pdf`
*   `lime_case_study_ip-spoofing_correct_classification_id22817.pdf`
*   `lime_case_study_ip-spoofing_misclassification_id12685.pdf`
*   `lime_case_study_manipulation_correct_classification_id13727.pdf`
*   `lime_case_study_manipulation_correct_classification_id19760.pdf`
*   `lime_case_study_mitm_correct_classification_id13771.pdf`
*   `lime_case_study_mitm_correct_classification_id19294.pdf`
*   `lime_case_study_password-cracking_correct_classification_id16340.pdf`
*   `lime_case_study_password-cracking_correct_classification_id23863.pdf`
*   `lime_case_study_replay_correct_classification_id1708.pdf`
*   `lime_case_study_replay_correct_classification_id8875.pdf`
*   `lime_case_study_unauth_correct_classification_id2323.pdf`
*   `lime_case_study_unauth_correct_classification_id8574.pdf`
*   `lime_case_study_video_correct_classification_id15087.pdf`
*   `lime_case_study_video_correct_classification_id20180.pdf`
