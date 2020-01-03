# Face detection, frontalization and recognition

A python library for improved face recognition in images. 

## Getting Started
### Prerequisites
numpy
matplotlib
torch
torchvision
glob
os
random
PIL
pathlib2
YouTubeFacesDB

## Usage and Examples

To train the networks images of format 64x64 have to be located in a folder named 'dataset'. The pipeline consists of three neural networks - one for face detection, one for face frontalization and one for face recognition.
While face detection and face recognition networks are both a pretrained vgg16 networks, the face frontalization network is implemented using pytorch.
The network has to be trained on the dataset. 
```
network = faceFront.FaceFront()
overfit_solver = s.Solver(optim=torch.optim.Adam,optim_args={"lr": 1e-4})
overfit_solver.train(network, traindata, num_epochs=5000, epochsize=100)
```
Now the network can frontalize a face on a picture.
```
a = torch.ones(64,64)
output = network(Variable(a.view(1,1,64,64)))
```

## Authors

This project is by

 * **Ritvik Ranadive** - [gitlab](https://gitlab.lrz.de/ga62maj)
 * **Armin Baur** - [gitlab](https://gitlab.lrz.de/ga38fun)
 * **Lena Hackl** - [github](https://github.com/lenamariahackl)
