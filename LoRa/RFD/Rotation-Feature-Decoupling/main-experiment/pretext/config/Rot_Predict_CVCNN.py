batch_size = 128

config = {}
# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = True
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'Lora'
data_train_opt['split'] = 'train'

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = True
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'Lora'
data_test_opt['split'] = 'val'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 300



networks = {}
LUT_lr = [(600,0.001)]
net_optim_params = {'optim_type': 'adam', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 0, 'nesterov': True, 'LUT_lr':LUT_lr}

# feature extractor network
networks['feature'] = {'def_file': 'architectures/CVCNNFeature.py', 'pretrained': None, 'opt': {},  'optim_params': net_optim_params}

# rotation classifier network
networks['classifier'] = {'def_file': 'architectures/CVCNNClassifier.py', 'pretrained': None, 'opt': {'num_classes':4},  'optim_params': net_optim_params}

# linear transformation normalization network
low_dim = 128
num_feat = 512
networks['norm'] = {'def_file': 'architectures/LinearTransformationNorm.py', 'pretrained': None, 'opt': {'low_dim':low_dim, 'num_feat':num_feat}, 'optim_params': net_optim_params}

config['networks'] = networks



criterions = {}

# Cross entropy loss
criterions['loss_cls'] = {'ctype':'CrossEntropyLoss', 'opt':{'reduce':False}}

# Distance loss
criterions['loss_mse'] = {'ctype':'MSELoss', 'opt':None}

# NCE average
ndata = 1281167
nceave_net_opt = {'low_dim':low_dim, 'ndata':ndata, 'nce_k':1024, 'nce_t':0.07, 'nce_m':0.5}
nceave_cri_opt = {'def_file': 'architectures/NCEAverage.py', 'net_opt':nceave_net_opt}
criterions['nce_average'] = {'ctype':'NCEAverage', 'opt':nceave_cri_opt}

# NCE criterion
ncecri_net_opt = {'ndata':ndata}
ncecri_cri_opt = {'def_file': 'architectures/NCECriterion.py', 'net_opt':ncecri_net_opt}
criterions['nce_criterion'] = {'ctype':'NCECriterion', 'opt':ncecri_cri_opt}

config['criterions'] = criterions

config['lambda_loss'] = {'cls':1.0, 'mse':1.0, 'nce':1.0}
config['gama'] = 2

config['algorithm_type'] = 'DecouplingModel'

config['best_metric']='loss_rot_mse'
