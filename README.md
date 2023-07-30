# GLARE: Accelerating Sparse DNN Inference Kernels with Global Memory Access Reduction
# Download the SDGC dataset
```
cd bin
./get_SDGC_dataset.sh --all
```
# Compile the executable
```
./compile.sh
```
# Run the executable
`-m`: method
`-n`: number of neurons per layer
`-l`: number of layers

To run benchmark 1024-120 on BF, run the following command in `bin/`.
```
./SDGC -m BF -n 1024 -l 120
```
To run method X with GLARE, please use `-m X_GLARE`. (Take BF as an example here).
```
./SDGC -m BF_GLARE -n 1024 -l 120
```
