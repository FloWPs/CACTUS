# CACTUS

[CACTUS multiview classifier for PWML Detection in Brain Volumes.]()

Introduced by Chen et al. (2021) the CrossViT is a type of Vision Transformer that uses a dual-branch architecture to extract multi-scale feature representations for image classification. The architecture combines image patches of different sizes to produce stronger visual features for image classification. A simple but effective token fusion module is proposed based on cross-attention, which uses a single token for each branch as a query to exchange information with the other branch.

In our work, we propose the Coronal and Axial Cross-ViT for Ultrasound (CACTUS), a model derived from the Cross-ViT architecture considering multi-view in addition to multi-scale patches in order to better combine information from different brain projections. Then we use this model to perform PWML and false alarm classification following a previous segmentation step. As a result, CACTUS will be applied to correct PWML predictions obtained at the output of the segmentation model.
