# Document-image-binarization-with-Cascaded-cGANs-generator

tools: 
-------
1.The python code is based on the python data science platform Anaconda3. 

2.The python code is tested on Windows by PyCharm.

3.Install PyTorch and dependencies from http://pytorch.org


Testing:
-------
1. few example test images are included in the testimg folder.

2. Run Test_All.py, test the binarization of an input image. 

3. Please download the pre-trained model from [here](https://pan.baidu.com/s/1Q1c19Sc7ubY7TATSZfKedw) , and put it under ./checkpoints/

4. [Executable file link](https://pan.baidu.com/s/1x6w_qqbK8lsHcmcnaUXT2Q) ：exe 

Training:
-------
1. Please download the public binarization datasets

2. Run combine_A_and_B_sub.py to get the sub training data.

3. Run train.py to train a new model. 


Acknowledgments:
-------
This code borrows heavily from pytorch-CycleGAN-and-pix2pix.
