**TITLE : Deep Learning-Based Detection of Canine Skin Diseases: Towards Accessible and Scalable Veterinary Diagnostics**

**DESCRIPTION :**
This repository contains the complete implementation of a deep learning pipeline for detecting six types of canine skin diseases using image-based classification.
The proposed framework combines Transfer Learning using Convolutional Neural Networks (CNN), Image Segmentation using Attention U-Net, Ensemble Learning using Stacking, and Explainability using Grad-CAM.
The pipeline is designed to produce a veterinary diagnostic support system that is accurate, scalable, reproducible, and clinically interpretable.
This work supports accessible and automated veterinary dermatology diagnostics, especially for environments with limited access to veterinary specialists.

**DATASET INFORMATION :**
The dataset was collected from multiple public sources including Kaggle and Roboflow Universe.

PRIMARY DATASET SOURCES :

Kaggle :

https://www.kaggle.com/datasets/youssefmohmmed/dogs-skin-diseases-image-dataset
https://www.kaggle.com/datasets/yashmotiani/dogs-skin-disease-dataset

Roboflow Universe :

https://universe.roboflow.com/dog-skin-disease-dermatosis/dog-skin-disease-dataset
https://universe.roboflow.com/kaivlya/animal-skin-disease-vb2io

DATASET CHARACTERISTICS :
Initial total images: ±5,740 images
Final dataset after preprocessing and segmentation: 5,816 images
Number of classes: 6 canine skin diseases
Image format: JPG / PNG
Disease Classes

The final dataset contains six diagnostic categories of canine skin diseases, including :
- Ringworm
- Demodicosis
- Dermatitis
- Bacterial Infection
- Fungal Infection
- Healthy Skin

(Note: final class naming may vary slightly depending on source dataset standardization.)

Data Processing Summary
- Multi-source dataset integration
- Duplicate removal (hash-based + manual verification)
- Irrelevant image filtering
- Label standardization
- Class balancing validation
- Ringworm lesion segmentation
- Final train/validation/test split
- Dataset Split
  Training Set: 70%
  Validation Set: 15%
  Testing Set: 15%

**CODE INFORMATION :**

This repository contains two main notebooks:

1. Segmentation_Ringworm.ipynb

This notebook performs lesion segmentation specifically for the Ringworm class using Attention U-Net.

Purpose :
Ringworm was selected because preliminary experiments showed it had the lowest classification accuracy due to subtle lesion visibility, high background dominance, inconsistent lesion appearance, and strong intra-class variability.

INPUT DATASET (SEGMENTATION) :
Use only the Ringworm folder from the raw dataset:
https://drive.google.com/drive/folders/1Yx9Q53DPpq9qDSJFPgDQ2xX2B-di9XjS?usp=sharing

Output :
- Binary lesion masks
- Overlay visualization
- Segmented ROI images for classification 2. dog_skin_disease_pred.ipynb

This notebook performs the full classification pipeline using the final combined dataset (including segmentation results).

Main Tasks :
Preprocessing
Augmentation
Transfer learning
CNN fine-tuning
Stacking ensemble
Performance evaluation
Confusion matrix generation
Grad-CAM explainability

INPUT DATASET (FOR TRAINING MODELS)
Use the final combined dataset:
https://drive.google.com/drive/folders/1DkTKkC-1n2C0c5hb0Bm0hILIqvz6JbKl?usp=sharing

Models Used :
- Custom CNN
- MobileNetV2
- DenseNet121
- ResNet50
- EfficientNetB0
  
Final Ensemble
Best-performing models:
- EfficientNetB0
- ResNet50

Combined using:
Stacking Ensemble Learning

Final test accuracy:
91.0%

**USAGE INSTRUCTIONS :**
Step 1 — Ringworm Segmentation

Open:
Segmentation_Ringworm.ipynb

Use only the Ringworm dataset from:

https://drive.google.com/drive/folders/1Yx9Q53DPpq9qDSJFPgDQ2xX2B-di9XjS?usp=sharing

Run all cells to generate:
- predicted masks
- lesion overlays
- segmented lesion images

Step 2 — Full Classification Pipeline

Open:
dog_skin_disease_pred.ipynb

Use the final combined dataset from:

https://drive.google.com/drive/folders/1DkTKkC-1n2C0c5hb0Bm0hILIqvz6JbKl?usp=sharing

Run all cells to:
- Preprocess images
- Train CNN models
- Fine-tune pretrained networks
- Build stacking ensemble
- Evaluate performance
- Generate Grad-CAM visualizations

Requirements :
Recommended environment:

- Python 3.10+
- Jupyter Notebook / Google Colab

Required Libraries :

pip install tensorflow

pip install keras

pip install numpy

pip install pandas

pip install matplotlib

pip install seaborn

pip install scikit-learn

pip install opencv-python

pip install pillow

pip install albumentations

pip install segmentation-models

pip install efficientnet

pip install grad-cam

pip install tqdm

Optional:
pip install kaggle
pip install roboflow

**METHODOLOGY :**

This study follows a structured Machine Learning Model Pipeline consisting of six major stages.

1. Data Gathering and Preparation
   - Collect image data from existing public datasets
   - Integrate additional datasets (up to 6 disease labels)
   - Clean data by removing duplicates and irrelevant samples
2. Data Validation
   - Check class balance (target maximum ratio 1:3)
   - Inspect image quality
   - Ensure label consistency after dataset merging
3. Data Preprocessing
   - Apply segmentation using Attention U-Net for Ringworm
   - Perform data augmentation
   - Normalize image dimensions and pixel values
4. Model Training and Tuning
   - Develop and fine-tune multiple pretrained CNN models
   - Implement two-stage fine-tuning
   - Develop a stacking ensemble classifier
   - Compare performance metrics to select the best configuration
5. Model Evaluation and Analysis
   - Evaluate model performance
   - Generate confusion matrix
   - Assess overfitting by comparing training and validation performance
6. Model Validation (Explainability)
   - Apply Grad-CAM to visualize important image regions
   - Interpret Grad-CAM results to confirm focus on disease areas

This methodology ensures robust, explainable, and clinically meaningful predictions for veterinary diagnostics.

**CITATIONS :**

[1] D. G. O’Neill, H. James, D. C. Brodbelt, D. B. Church, C. Pegram, Prevalence of commonly diagnosed disorders in uk dogs under primary veterinary care: Results and applications, BMC Veterinary Research 17 (2021). doi:10.1186/s12917-021-02775-3.

[2] A. Zahri, Y. Zair, R. Jawabri, S. Elhansa, Y. Lahlou, M. Ziyadi, M. Aouissi, M. Elhachimi, Survey on dermatological disorders of dogs during 2020–2022 in rabat, mo- rocco, World Veterinary Journal 14 (3) (2024). doi:10.54203/scil.2024.wvj52.

[3] R. Marsella, Advances in our understanding of canine atopic dermatitis, Veterinary Dermatology 32 (2021). doi:10.1111/vde.12965.

[4] Y. Dong, L. Wang, K. Zhang, H. Zhang, D. Guo, Prevalence and association with environmental factors and establishment of prediction model of atopic dermatitis in pet dogs in china, Frontiers in Veterinary Science 11 (2024) 1428805. doi:10.3389/fvets. 2024.1428805.

[5] M. Fragoso-García, S. Sura, K. Wakamatsu, T. Akiyama, Y. Murofushi, R. Ogasawara, Automated diagnosis of 7 canine skin tumors using machine learning on h&e-stained whole slide images, Veterinary Pathology 60 (6) (2023). doi:10.1177/03009858231189205.

[6] A. Smith, L. Miller, K. Athanasiadis, S. Baines, Computer vision model for the detection of canine pododermatitis and neoplasia of the paw, Veterinary Dermatology 35 (2) (2024). doi:10.1111/vde.13221.

[7] A. D. Paryuni, S. Indarjulianto, S. Widyarini, Dermatophytosis in companion animals: A review, Veterinary World 13 (6) (2020) 1174–1181. doi:10.14202/vetworld.2020.1174-1181.

[8] L. Alzubaidi, J. Zhang, A. J. Humaidi, A. Al-Dujaili, Y. Duan, O. Al-Shamma, J. Santamaría, M. A. Fadhel, M. Al-Amidie, L. Farhan, Review of deep learning: Concepts, cnn architectures, challenges, applications, future directions, Journal of Big Data 8 (2021). doi:10.1186/s40537-021-00444-8.

[9] S. Xiao, N. K. Dhand, Z. Wang, K. Hu, P. C. Thomson, J. K. House, M. S. Khatkar, Review of applications of deep learning in veterinary diagnostics and animal health, Frontiers in Veterinary Science 12 (2025) 1511522. doi:10.3389/fvets.2025.1511522.

[10] P. Ezanno, S. Picault, G. Beaunée, C. Legeay, C. Goursaud, C. Quiniou, B. Audouin, C. Burel, J.-M. Denoix, S. Falala, N. Grimaud, D. Rault, M. Rault, L. Reverte, C. Saegerman, B. Sallé, M. Tretout, F. Zagmutt, O. Zongo, Research perspectives on animal health in the era of artificial intelligence, Veterinary Research 52 (2021). doi:10.1186/s13567-021-00902-4.

[11] A. Upadhyay, G. Singh, S. Mhatre, P. Nadar, Dog skin diseases detection and iden- tification using convolutional neural networks, SN Computer Science 4 (4) (2023). doi:10.1007/s42979-022-01645-5.

[12] H. Yu, I.-G. Lee, J.-Y. Oh, J. Kim, J.-H. Jeong, K. Eom, Deep learning-based ultrasono- graphic classification of canine chronic kidney disease, Frontiers in Veterinary Science 11 (2024). doi:10.3389/fvets.2024.1443234.

[13] S. Hwang, H. K. Shin, J. M. Park, B. Kwon, M.-G. Kang, Classification of dog skin diseases using deep learning with images captured from a multispectral imaging device, Journal of Sensor Science and Technology 31 (6) (2022). doi:10.5578/jsts.2022.31. 6.455.

[14] I. Abunadi, E. M. Senan, Deep learning and machine learning techniques for diagnosis of dermoscopy images for early detection of skin diseases, Electronics 10 (24) (2021). doi:10.3390/electronics10243158.

[15] K. Ali, Z. A. Shaikh, A. A. Khan, A. A. Laghari, Multiclass skin cancer classification using efficientnets – a first step towards preventing skin cancer, Neuroscience Informatics 2 (4) (2021). doi:10.1016/j.neuri.2021.100034.

[16] M. Sharma, B. Jain, C. Kargeti, V. Gupta, D. Gupta, Detection and diagnosis of skin diseases using residual neural networks (resnet), Journal of Engineering, Design & Technology (2021). doi:10.1142/S0219467821400027.

[17] A. R. Ajel, A. Q. Al-Dujaili, Z. G. Hadi, A. J. Humaidi, Skin cancer classifier based on convolution residual neural network, International Journal of Electrical and Computer Engineering 13 (6) (2023). doi:10.11591/ijece.v13i6.pp6240-6248.

[18] P. E. N. Taruno, G. S. Nugraha, R. Dwiyansaputra, F. Bimantoro, Monkeypox classification based on skin images using cnn: Efficientnet-b0, E3S Web of Conferences 465 (2023). doi:10.1051/e3sconf/202346502031.

[19] M. M. Ahsan, M. R. Uddin, M. S. Ali, M. K. Islam, M. Farjana, A. N. Sakib, K. Al Momin, S. A. Luna, Deep transfer learning approaches for monkeypox disease diagnosis, Expert Systems with Applications 216 (2023). doi:10.1016/j.eswa.2022. 119483.

[20] Y. Gulzar, S. Agarwal, S. Soomro, M. Kandpal, S. Turaev, W. Onn Choo, S. Saini, A. Bounsiar, Next-generation approach to skin disorder prediction employing hybrid deep transfer learning, Frontiers in Big Data 8 (2025). doi:10.3389/fdata.2025. 1503883.

[21] D. W. Girmaw, Livestock animal skin disease detection and classification using deep learning approaches, Biomedical Signal Processing and Control 102 (2025). doi:10.1016/j.bspc.2024.107334.

[22] T. Adilah M, D. A. Kristiyanti, Implementation of transfer learning mobilenetv2 architecture for identification of potato leaf disease, Journal of Theoretical and Applied Information Technology 101 (16) (2023).

[23] R. A. Pratama, Implementasi ensemble deep learning untuk klasifikasi penyakit pada tanaman padi (2023).

[24] J. Alanazi, Detection of skin lesions in dogs using advanced convolutional neural network technology, Indian Journal of Animal Research (2025). doi:10.18805/IJAR.BF-1820.

[25] A. A. AlZubi, M. Al-Zu’bi, Application of artificial intelligence in monitoring of animal health and welfare, Indian Journal of Animal Research 57 (11) (2023). doi:10.18805/ IJAR.BF-1698.

[26] M. Fiaz, M. B. Shoaib Khan, A. H. Khan, A. Bilal, M. Abdullah, A. A. Darem, R. Sarwar, Correction: An explainable hybrid deep learning framework for precise skin lesion segmentation and multi-class classification, Frontiers in Medicine 12 (2025). doi:10.3389/fmed.2025.1724427.

[27] W. Bakasa, S. Viriri, Stacked ensemble deep learning for pancreas cancer classification using extreme gradient boosting, Frontiers in Artificial Intelligence 6 (2023). doi:10.3389/frai.2023.1232640.

[28] M. Khaled, D. Gaceb, F. Touazi, C. A. Aouchiche, Y. Bellouche, A. Titoun, New cnn stacking model for classification of medical imaging modalities and anatomical organs on medical images, in: International Workshop on Informatics & Data-Driven Medicine, 2023. URL:https://ceur-ws.org/Vol-3609/paper14.pdf

[29] Y. Sun, J. Guo, Y. Liu, N. Wang, Y. Xu, F. Wu, J. Xiao, Y. Li, X. Wang, Y. Hu, Y. Zhou, Metnet: A novel deep learning model predicting met dysregulation in non- small-cell lung cancer on computed tomography images, Computers in Biology and Medicine 171 (2024). doi:10.1016/j.compbiomed.2024.108136.

[30] S. Burti, T. Banzato, S. Coghlan, M. Wodzinski, M. Bendazzoli, A. Zotti, Artificial intelligence in veterinary diagnostic imaging: Perspectives and limitations, Research in Veterinary Science 175 (2024). doi:10.1016/j.rvsc.2024.105317.

[31] H. Suksangvoravong, N. Choisunirachon, T. Tongloy, S. Chuwongin, S. Boonsang, V. Kittichai, C. Thanaboonnipat, Automatic classification and grading of canine tra- cheal collapse on thoracic radiographs by using deep learning, Veterinary Radiology & Ultrasound 65 (6) (2024). doi:10.1111/vru.13413.

[32] M. Hubbard-Perez, A. Luchian, C. Milford, L. Ressel, Use of deep learning for the classification of hyperplastic lymph node and common subtypes of canine lymphomas: a preliminary study, Frontiers in Veterinary Science 10 (2024). doi:10.3389/fvets. 2023.1309877.

[33] Mohamed, Y. Dog's Skin Diseases (Image Dataset). Kaggle. Retrieved September 13, 2025, from https://www.kaggle.com/datasets/youssefmohmmed/dogs-skin-diseases-image-dataset

[34] Motiani, Y. Dogs Skin Disease Dataset. Kaggle. Retrieved September 13, 2025, from https://www.kaggle.com/datasets/yashmotiani/dogs-skin-disease-dataset

[35] Kaivlya. Animal Skin Disease Computer Vision Dataset. Roboflow Universe. Retrieved September 13, 2025, from https://universe.roboflow.com/kaivlya/animal-skin-disease-vb2io

[36] Dog Skin Disease Dermatosis. Dog Skin Disease Dataset Computer Vision Model. Roboflow Universe. Retrieved September 13, 2025, from https://universe.roboflow.com/dog-skin-disease-dermatosis/dog-skin-disease-dataset
