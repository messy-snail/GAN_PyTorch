# GAN_PyTorch
Various GAN implementations based on PyTorch. This project is consist of simple and standard version. The Simple version has a relatively short code length, and only simple functions are implemented.   
The Standard version has various functions rather than the simple version. It also provides a UI using PyQt(In this case, the standard version is loaded and executed).
~~In fact, I don't know if UI is comfortable...~~


### Experiment Environment
* Windows 10 Enterprise
* Intel i7-3770k
* RAM 12.0 GB
* NVIIDA GTX TITAN
* PyTorch 0.4.0
* torchvision 0.2.1
* PyQt 5


### Vanilla_GAN  
MLP-based regular GAN is implemented. Ian Goodfellow's paper used Maxout, ReLU, and SGD. But the performance is not working properly, so I modified it and implemented it.   
[Paper](https://arxiv.org/pdf/1406.2661.pdf)
#### Vanilla_Simple.py
* This is a brief implementation of the Vanilla GAN, and the functions are described below by block.  
![res_code](https://user-images.githubusercontent.com/38720524/42674458-c5a45f7a-86aa-11e8-9b73-0a8d26f01610.png)
* This code uses the MNIST data set.

#### **Import**
**Import the necessary libraries.**
* torch : Library to implement tensor or network structures
* torchvision : Library for managing datasets
* os : Library for loading file path
```python
import torch as tc
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
```
#### **Parameter**
**Set the image size, result path, and hyper parameter for learning.**
* result_path : Path where the results are saved.
* img_sz : Image size.(MNIST =28)
* noise_sz : Latent code size which is the input of generator.
* hidden_sz : Hidden layer size.(The number of nodes per hidden layer)
* batch_sz : Batch size.
* nEpoch : Epoch number.
* nChannel : Channel size.(MNIST=1)
* lr : Learning rate.  
```python
result_path = 'simple'
img_sz = 784
noise_sz = 100
hidden_sz = 512
batch_sz = 100
nEpoch = 300
nChannel = 1
lr = 0.0002
```
#### **Data load**
**Load the dataset. This project used MNIST dataset.**
* trans : Transform the dataset.
  * `Compose()`is used when there are multiple transform options. Herem `ToTensor()` and `Normalize(mean, std)` are used.
  * `ToTensor ()` changes the PIL Image to a tensor. torchvision dataset The default type is PIL Image.
  * `Normalize (mean, std)` transforms the range of the image. Here, the value of [0, 1] is adjusted to [-1, 1]. ((value-mean) / std)

* dataset : Store (MNIST data) at the specified location.
  * root : This is the path to store (MNIST data). Folders are automatically created with the specified name.
  * train : Set the data to be used for the train.
  * transform : Transform the data according to the transform option set previously.
  * download : Download (MINST data). (If you downloaded it once, it will not do it again.)

* dataloader : Load the data in the dataset.
  * dataset : Set the dataset to load.
  * batch_size : Set the batch size. 
  * shuffle : Shuffle the data and load it.

```python
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = torchvision.datasets.MNIST(root='./MNIST_data', train=True, transform=trans, download=True)
dataloader = tc.utils.data.DataLoader(dataset=dataset, batch_size=batch_sz, shuffle=True)
```

#### **Range**
**[0, 1] in the range of [-1, 1].**
* Clamp changes the value of 0 or less to 0, and the value of 1 or more to 1.

```python
def img_range(x):
    out = (x+1)/2
    out = out.clamp(0, 1)
    return(out)
```

#### **Discriminator**
**Create a Discriminator**
* Sigmoid was placed on the last layer to output [0, 1]. (0 : Fake, 1 : Real)
```python
D = nn.Sequential(
    nn.Linear(img_sz, hidden_sz),
    nn.ReLU(),
    nn.Linear(hidden_sz, hidden_sz),
    nn.ReLU(),
    nn.Linear(hidden_sz, 1),
    nn.Sigmoid()
)
```

#### **Generator**
**Create a Generator**
* Tanh is placed on the last layer to output [-1, 1].
```python
G = nn.Sequential(
    nn.Linear(noise_sz, hidden_sz),
    nn.ReLU(),
    nn.Linear(hidden_sz, hidden_sz),
    nn.ReLU(),
    nn.Linear(hidden_sz, img_sz),
    nn.Tanh()
)
```

#### **GPU**
**Pass the network to the GPU.**
* If `is_available ()` is true, the GPU is used. If it is false, CPU is used. 
```python
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
D = D.to(device)
G = G.to(device)
```

#### **Optimizer**
**Set the optimizer to optimize the loss function.**
* Loss function is set to `BCELoss ()` and Binary Cross Entropy Loss. The definition of BCE is `BCE (x, y) = -y * log (x) - (1-y) * log (1-x)`.  

```python
loss_func = tc.nn.BCELoss()
d_opt = tc.optim.Adam(D.parameters(), lr=lr)
g_opt = tc.optim.Adam(G.parameters(), lr=lr)
```

#### **Training**
**The training process consists of learning the discriminator and learning the generator.**
##### **Train the D**
* Load the images from the dataloader  
![res_images](https://user-images.githubusercontent.com/38720524/42674466-cc89e49a-86aa-11e8-97a3-4d49bec60c6c.png)
* Flatten the images in one dimension to fit MLP.  
![res_reshape](https://user-images.githubusercontent.com/38720524/42674483-e5b43542-86aa-11e8-99b6-1daa05b0c7d1.png)
* Generate noise (lantic code) for the input of the generator.  
![res_noise](https://user-images.githubusercontent.com/38720524/42674500-f4b12898-86aa-11e8-843b-b99ecaafc8c3.png)
* Create a label for discriminator learning.  
![res_label](https://user-images.githubusercontent.com/38720524/42674512-01e5ec06-86ab-11e8-9020-136d24046aa5.png)
* In Discriminator, Input the images and the fake images (G (z)). Find the loss function using labels (real: 1, fake: 0).  
![res_loss](https://user-images.githubusercontent.com/38720524/42674520-09f47868-86ab-11e8-8190-61abe0c85a8b.png)
* Add each loss to find the total loss, and use the `backward ()` function to find the gradient of each node. `step ()` updates the parameters(w,b) according to the optimizer option defined above. **Note that only the discriminator is learned.**  
![res_update](https://user-images.githubusercontent.com/38720524/42674527-11d30f36-86ab-11e8-8657-9edce41ecb54.png)
```python
for ep in range(nEpoch):
    for step, (images, _) in enumerate(dataloader):
        images = images.reshape(batch_sz, -1).to(device)
        z = tc.randn(batch_sz, noise_sz).to(device)

        real_label = tc.ones(batch_sz, 1).to(device)
        fake_label = tc.zeros(batch_sz, 1).to(device)

        loss_real = loss_func(D(images), real_label)
        loss_fake = loss_func(D(G(z)), fake_label)

        d_loss = loss_real + loss_fake

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()
```
#### **Train the G**
* Perform the learning in a similar way as before. **Note that only learn about the generator.**  
![res_g_train](https://user-images.githubusercontent.com/38720524/42674532-1af03e86-86ab-11e8-8d1e-db360a3bf58d.png)
```python
        fake_images = G(z)
        g_loss = loss_func(D(fake_images), real_label)

        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()
```
##### **Log & Image save**
* Print the log and seve the image.
```python
        if step%200 ==0:
            print('epoch {}/{}, step {}, d_loss {:.4f}, g_loss {:.4f}, Real_score {:.2f}, Fake_score {:.2f}'.format(ep, nEpoch, step+1, d_loss.item(), g_loss.item(), D(images).mean().item(), D(fake_images).mean().item()))

    if ep==0:
        out = images.reshape(mini, nChannel, img_sz, img_sz)
        out = img_range(out)
        save_image(out, os.path.join(result_path, 'real_img.png'))
    out = fake_images.reshape(mini, nChannel, img_sz, img_sz)
    out = img_range(out)
    save_image(out, os.path.join(result_path, 'fake_img {}.png'.format(ep)))
```
#### **Result**
* The figure below shows the results as the epoch increases.
![fake_image1](https://user-images.githubusercontent.com/38720524/42674543-25791a4e-86ab-11e8-8e1d-ca33475c6bb2.png)
![fake_image15](https://user-images.githubusercontent.com/38720524/42674545-25a1a7c0-86ab-11e8-83da-9199d8f5d12a.png)
![fake_image60](https://user-images.githubusercontent.com/38720524/42674546-25c73c42-86ab-11e8-8081-0cbccb2bd2d8.png)
![fake_image1000](https://user-images.githubusercontent.com/38720524/42674547-25ecf748-86ab-11e8-8c5b-ad28f15daaa5.png)


#### Vanilla_Standard.py & for_UI.py
will be updated