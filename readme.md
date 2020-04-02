##About

This is a project for the Deploying a machine learning model on edge devices such as intel's neural compute stick. We tried deploying a siamese neural network for face frontal face recognition. 

The objective of this project is to experiment the deployment of a working model on an edge device. Owning to the applicability of face recognition, we tried exploring the performance of the deployment of a quantized deep learning model on an edge device like  Intel's Neural compute stick. 

We explore this using Viola jones face reconginition combined with the siamese neural network trained on predefined datase quantized and optimized with further to run on intel processor.The experimentation resulted in similar frame per second performance for running a quantized model on a edge device like movidius ncs and intel i7 2nd generation laptop. 

## Getting started

## Code for training Siamese Neural Network:

```sh
Training(https://github.com/Shashankwer/SiameseNCS/blob/master/Scripts/Siamese_NCS.ipynb)
```

## Deployment

Refer to instructions on deployment.txt(https://github.com/Shashankwer/SiameseNCS/blob/master/Deployment%20Steps.txt)

##Architecture description

Reference to the pdf(https://github.com/Shashankwer/SiameseNCS/blob/master/NCS_Report.pdf)
 
### Results on CPU

Refer to file(https://github.com/Shashankwer/SiameseNCS/blob/master/output_cpu.mp4)

### Results on Movidius

Refer to file(https://github.com/Shashankwer/SiameseNCS/blob/master/output_myriad.mp4)
