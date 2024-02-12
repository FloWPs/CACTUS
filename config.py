from easydict import EasyDict

cfg = EasyDict()

cfg.data_dir = './dataset/'
cfg.volumes = './meta_patient_T1_FLAIR_SPM12.npy'
cfg.gt = './meta_patient_ground_truth_T1_FLAIR_SPM12.npy'
cfg.brain_seg = './meta_patient_seg_brain_T1_FLAIR_SPM12.npy'

cfg.train_patlist = [i for i in range(60)]
cfg.test_patlist = [i for i in range(60, 170)]

#----------------
#    Training
#----------------

cfg.model_path = 'D:/CACTUS/models/'
cfg.kfold = 2
cfg.batch_size = 32
cfg.epoch = 100
cfg.learning_rate = 1e-3
cfg.patience = 10

cfg.cactus = EasyDict()
cfg.cactus.image_size = 32
cfg.cactus.channels = 3
cfg.cactus.num_classes = 2
cfg.cactus.patch_size_axial = 2
cfg.cactus.patch_size_coronal = 2
cfg.cactus.axial_dim = 24
cfg.cactus.coronal_dim = 24
cfg.cactus.axial_depth = 1
cfg.cactus.coronal_depth = 1
cfg.cactus.cross_attn_depth = 1
cfg.cactus.multi_scale_enc_depth = 3
cfg.cactus.heads = 3

#------------------
#    Inference
#------------------

cfg.inference = EasyDict()
cfg.inference.model_path = './models/2024-2-8_11-27/config0/cactus_config0.pth'
cfg.inference.projection = 'coronal'
cfg.inference.threshold = 0.6
cfg.inference.min_lesion_size = 50
cfg.inference.max_lesion_size = 10000 # optional
