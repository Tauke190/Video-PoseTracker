mkdir -p ~/datasets/posetrack
cd ~/datasets/posetrack

# Base URL
BASE="https://hyper.ai/en/datasets/5729/download?path=/PoseTrack/data/PoseTrack2018"

# Download all PoseTrack2018 image parts
for part in aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq ar; do
  wget "${BASE}/posetrack18_images.tar.${part}" -O "posetrack18_images.tar.${part}"
done

# Download labels
wget "${BASE}/posetrack18_v0.45_public_labels.tar.gz" -O "posetrack18_v0.45_public_labels.tar.gz"
