
# DFAFN

# Breast Cancer HER2 Status Recognition Based on Dual-Modal Feature Attention Fusion Network

# Proposed method

The DFAFN is composed of a HER2 Ultrasound Image Feature Extractor (HUFE), a HER2 Radiomic Feature Extractor (HRFE), a HER2 Status-related Feature Learning

(HSFL) Module, a Two-step Cross-Attention Fusion (TCAF) Network, and a Fully Connected Classification Module. Firstly, we conducted image preprocessing to facilitate the extraction of deep features and radiomic features from ultrasound images. We then employed a HER2 Ultrasound Image Feature Extractor and a HER2 Radiomic Feature Extractor to

obtain the deep features and HER2 radiomic features, respectively. Through the HER2 Status-related Feature Learning Module, we extracted features that are closely associated

with HER2 status, obtaining relevant features from both modalities. Next, we designed a Two-step Cross-Attention Fusion Network that utilizes an attention mechanism

to integrate these two sets of features, resulting in fused features with enhanced interactions. Finally, the fused features are fed into a Fully Connected Classification

Module to complete the HER2 status classification task. A detailed explanation of the method is illustrated in the Fig. 1.

The figure below shows our proposed network.

<img width="1710" height="920" alt="image" src="https://github.com/user-attachments/assets/b74a682b-47aa-4b63-8d5b-91734b52a440" />


Install dependencies

Train by yourself

If you want to train by yourself, you can run this command :
    python run DFAFN.py



# Breast Cancer HER2 Status Recognition Based on Dual-Modal Feature Attention Fusion Network

# Proposed method

