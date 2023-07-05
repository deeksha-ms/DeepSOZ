Training scripts for all models.

Proposed model DeepSOZ: "txlstm_szpool"

Baselines: "txlstm_maxpool", "sztrack", "cnnblstm", "szloc"

TRAINING procedure: (nested cross validation) Pretrain > Retrain and then finetune. For more details, kindly refer to our manuscript.   

Prerequisited:

Pytorch 1.9.0 cuda 11.1
(pip install --user torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html)

Numpy

Pandas

Preprocessed and windowed EEG data and  manifest files stored in multiple subfolds
