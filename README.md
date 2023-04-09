# cuda-examples

Max Vector Entries = num vectors * vector scalar * dims * bytes = 3 * 2000*60*128 * 4 ~ 0.185 GB


Max Matrix Entries = num matrices * matrix scalar * dims * bytes = 3 * 32*1024*1024 * 4 ~ 0.402 GB


Setting up GCP -

1) Create new VM Instance
   a) navigate to create new VM Instance
   b) under Machine configuration, select GPUs. GPU type can be NVIDIA T4; Machine type can be n1-standard-2 (2 vCPU, 7.5 GB memory).. our largest program is the 1024-dim matrix, which will use approximately > 0.4 GB of memory
   c) under Boot disk, click CHANGE. under Public Images, choose "Deep Learning on Linux" from the Operating System drop down, and then "Debian 10 based Deep Learning VM with M106". This will come with CUDA 11.3 installed
NOTE - GCP has an annoying tendency to not have available machines in a region, so you may need to do this a couple of times and change the region until you find one with machines

2) SSH into the new VM
   a) it will ask if you want to install NVIDIA drivers when you first log in- do this


2) Git should be available on the VM