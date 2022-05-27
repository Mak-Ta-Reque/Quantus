#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  
  # put your install commands here (remove lines you don't need):
  #pip install torch==1.7.0+cu102 torchvision==0.9.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
  #pip install apex -f https://dl.fbaipublicfiles.com/vissl/packaging/apexwheels/py36_cu102_pyt170/download.html
  #git clone --recursive https://github.com/Mak-Ta-Reque/vissl.git
  #cd vissl/
  #git checkout v0.1.6
  #git checkout -b v0.1.6
  conda install pip
  conda install -r requirements.txt
  #pip install opencv-python
  #pip uninstall -y classy_vision
  #pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d
  #pip uninstall -y fairscale
  #pip install fairscale@https://github.com/facebookresearch/fairscale/tarball/df7db85cef7f9c30a5b821007754b96eb1f977b6
  #pip install -e .[dev]
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi
