import torch as tc
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

#set a result path
result_path = 'result'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

img_sz = 28
noise_sz = 100
hidden_sz = 512
batch_sz = 128
nEpoch = 300
nChannel = 1
lr = 0.0002

#Data load
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = torchvision.datasets.MNIST(root='./MNIST_data', train=True, transform=trans, download=True)
dataloader = tc.utils.data.DataLoader(dataset=dataset, batch_size= batch_sz, shuffle= True)

#De-normalize
def img_range(x):
    out = (x+1)/2
    out = out.clamp(0, 1)
    return(out)

#Discriminator
#LeakyReLU is technique of DCGAN. This activation function is good in Discriminator.
D = nn.Sequential(
    nn.Linear(img_sz*img_sz*nChannel, hidden_sz),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_sz, hidden_sz),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_sz, 1),
    nn.Sigmoid()
)

#Generator
G = nn.Sequential(
    nn.Linear(noise_sz, hidden_sz),
    nn.ReLU(),
    nn.Linear(hidden_sz, hidden_sz),
    nn.ReLU(),
    nn.Linear(hidden_sz, img_sz*img_sz*nChannel),
    nn.Tanh()
)

#Device
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
D = D.to(device)
G = G.to(device)

loss_func = tc.nn.BCELoss()
d_opt = tc.optim.Adam(D.parameters(), lr=lr)
g_opt = tc.optim.Adam(G.parameters(), lr=lr)

for ep in range(nEpoch):
    # Read Image
    for step, (images, a) in enumerate(dataloader):
        mini = images.size()[0]
        images = images.reshape(mini, -1).to(device)
        z = tc.randn(mini, noise_sz).to(device)

        real_label = tc.ones(mini, 1).to(device)
        fake_label = tc.zeros(mini, 1).to(device)

        #D train
        loss_real = loss_func(D(images), real_label)
        loss_fake = loss_func(D(G(z)), fake_label)

        d_loss = loss_real + loss_fake

        D.zero_grad()
        d_loss.backward()
        d_opt.step()

        #G train
        z = tc.randn(mini, noise_sz).to(device)
        fake_images = G(z)
        g_loss = loss_func(D(fake_images), real_label)

        G.zero_grad()
        g_loss.backward()
        g_opt.step()

        if step%200 ==0:
            print('epoch {}/{}, step {}, d_loss {:.4f}, g_loss {:.4f}, Real_score {:.2f}, Fake_score {:.2f}'.format(ep, nEpoch, step+1, d_loss.item(), g_loss.item(), D(images).mean().item(), D(fake_images).mean().item()))

    if ep==0:
        out = images.reshape(mini, nChannel, img_sz, img_sz)
        out = img_range(out)
        save_image(out, os.path.join(result_path, 'real_img.png'))
    out = fake_images.reshape(mini, nChannel, img_sz, img_sz)
    out = img_range(out)
    save_image(out, os.path.join(result_path, 'fake_img {}.png'.format(ep)))

    tc.save(G.state_dict(), 'G.ckpt')
    tc.save(D.state_dict(), 'D.ckpt')