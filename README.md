# CACTUS

[CACTUS multiview classifier for PWML Detection in Brain Volumes.](https://ieeexplore.ieee.org/document/10793934)

Punctate white matter lesions (PWML) are the most common white matter injuries found in preterm neonates, with several studies indicating a connection between these lesions and negative long-term outcomes.

__Automated detection of PWML through cranial ultrasound (cUS) imaging__ could assist doctors in diagnosis more effectively and at a lower cost than MRI. However this task is highly challenging because of the lesions' small size and low contrast, and the number of lesions can also vary significantly between subjects.

In this work, we propose a two-phase approach:
  1) Segmentation using a vision transformer to increase the detection rate of lesions.
  2) __Multi-view classification leveraging cross-attention to reduce false positives and enhance precision with CACTUS__ model. We also investigate multiple postprocessing approaches to ensure prediction quality and compare our results with what is known in MRI.

Our method demonstrates improved performance in PWML detection on cUS images, achieving recall and precision rates of 0.84 and 0.70, respectively, representing an increase of 2% and 10% over the best published cUS models. Moreover, by reducing the task to a slightly simpler problem (detection of MRI-visible PWML), the model achieves 0.82 recall and 0.89 precision, which is equivalent to the latest method in MRI.
