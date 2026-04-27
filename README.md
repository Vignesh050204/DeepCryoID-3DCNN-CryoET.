# DeepCryoID 🔬

A deep learning-based pipeline for automated identification and 
classification of protein complexes in Cryo-Electron Tomography 
(CryoET) subtomogram images using 3D Convolutional Neural Networks.

## 🎯 Overview

Cryo-ET produces 3D volumetric images of cellular structures, but 
manual classification is time-consuming and error-prone. DeepCryoID 
automates this process using a custom 3D-CNN architecture trained on 
the SHREC 2019 benchmark dataset.

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 91% - 97% |
| Dataset | SHREC 2019 |
| Classes | 10 protein complexes |

## 🛠️ Tools & Technologies

- **Deep Learning:** PyTorch
- **Language:** Python 3
- **Dataset:** SHREC 2019 CryoET Benchmark
- **UI:** CustomTkinter (Desktop App), Flask (Web App)
- **Notebook:** Jupyter Notebook

- ## 📁 Project Structure
- DeepCryoID-3DCNN-CryoET/
│
├── full_dataset/                  # SHREC 2019 dataset
├── Desktop_app.py                 # CustomTkinter desktop app
├── protein_project_v2.ipynb       # Main training notebook
├── protein_model.pth              # Trained model weights
│
├── graph_1_sample_volumes.png     # Sample CryoET volumes
├── graph_2_training_curves.png    # Training & validation curves
├── graph_3_confusion_matrix.png   # Confusion matrix
├── graph_4_metrics.png            # Performance metrics
├── graph_5_architecture.png       # 3D-CNN architecture
├── graph_6_dashboard.png          # Results dashboard
├── graph_7_live_prediction.png    # Live prediction output

