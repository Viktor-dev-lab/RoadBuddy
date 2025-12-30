

# Download TransnetV2
git clone https://github.com/soczech/TransNetV2.git

Get stattion at link: My Project\RoadBuddy\third_party\TransNetV2\inference


# Tải file saved_model.pb
wget -O transnetv2-weights/saved_model.pb https://github.com/soczech/TransNetV2/raw/master/inference/transnetv2-weights/saved_model.pb
# Tải các file variables
wget -O transnetv2-weights/variables/variables.data-00000-of-00001 https://github.com/soczech/TransNetV2/raw/master/inference/transnetv2-weights/variables/variables.data-00000-of-00001
wget -O transnetv2-weights/variables/variables.index https://github.com/soczech/TransNetV2/raw/master/inference/transnetv2-weights/variables/variables.index


# Download Sam2
cd "D:\Workstation\My Project\RoadBuddy\third_party"

# Clone repo đầy đủ (bao gồm configs)
git clone https://github.com/facebookresearch/segment-anything-2.git SAM2

# Cài đặt
cd SAM2
pip install -e .

# Download checkpoint
mkdir checkpoints
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" -OutFile "sam2.1_hiera_tiny.pt"

# DeepSort tracking
deep-sort-realtime>=1.3.2