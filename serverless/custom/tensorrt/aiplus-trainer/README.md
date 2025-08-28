download TensorRT-8.6.0.12.Linux.x86_64-gnu.cuda-11.8.tar.gz into the assets folder

```bash
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.0/tars/TensorRT-8.6.0.12.Linux.x86_64-gnu.cuda-11.8.tar.gz
```

then deploy:

```bash
nuctl deploy aiplus-trainer --path ./serverless/custom/tensorrt/aiplus-trainer --project-name cvat --namespace nuclio --platform local
```