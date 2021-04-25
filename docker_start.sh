# start tensorboard and http server in background
echo "DeepCoMP: Starting TensorBoard and HTTP Server (for videos)..."
mkdir results
mkdir results/train
cd results
python3 -m http.server &
tensorboard --logdir train/ --host 0.0.0.0
