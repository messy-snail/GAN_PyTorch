import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

#set a result path
result_path = 'result'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

latent_sz = 100
batch_sz = 128
epoch_sz = 30
lr = 0.0002
img_sz = 64

trans = transforms.Compose([transforms.Resize((img_sz, img_sz)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#LSUN
#dataset = tv.datasets.LSUN('.', classes=['bedroom_train'], transform=trans)
#CelebA
dataset = tv.datasets.ImageFolder('./img_align_celeba', trans)
dataloader = tc.utils.data.DataLoader(dataset=dataset, batch_size= batch_sz, shuffle= True)

def image_range(x):
    out = (x+1)/2
    out = out.clamp(0,1)
    return(out)

#weight and bias Initialization
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, latent_sz):
        super(Generator, self).__init__()
        self.tconv1 = nn.ConvTranspose2d(latent_sz, 1024, 4, 1, 0)
        self.tconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.tconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.tconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.tconv5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

        self.bn1 = nn.BatchNorm2d(1024)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, input):
        x = F.relu(self.bn1(self.tconv1(input)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))
        x = F.tanh(self.tconv5(x))

        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.conv5 = nn.Conv2d(1024, 1, 4, 1, 0)

        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

#D, G 생성
G = Generator(latent_sz)
D = Discriminator()

G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)

#Module cuda

device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

G.to(device)
D.to(device)

loss_func = tc.nn.BCELoss()
g_opt = tc.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999)) #0.999
d_opt = tc.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
print("Processing Start")
for ep in range(epoch_sz):
    for step, (images, _) in enumerate(dataloader):
        images = images.to(device)
        mini_batch = images.size()[0]
        z = tc.randn(mini_batch, latent_sz).view(-1, latent_sz, 1, 1).to(device)

        real_label = tc.ones(mini_batch).to(device)
        fake_label = tc.zeros(mini_batch).to(device)

        D_result = D(images).squeeze()
        loss_real = loss_func(D_result, real_label)
        D_result = D(G(z)).squeeze()
        loss_fake = loss_func(D_result, fake_label)

        d_loss = loss_real+loss_fake
        D.zero_grad()
        d_loss.backward()
        d_opt.step()

        z = tc.randn(mini_batch, latent_sz).view(-1, latent_sz, 1, 1).to(device)
        fake_images = G(z)
        D_result = D(fake_images).squeeze()
        g_loss = loss_func(D_result, real_label)

        G.zero_grad()
        g_loss.backward()
        g_opt.step()

        if step%200 ==0:
            print('epoch {}/{}, step {}, d_loss {:.4f}, g_loss {:.4f}, Real_score {:.2f}, Fake_score {:.2f}'.format(ep, epoch_sz, step, d_loss.item(), g_loss.item(), D(images).mean().item(), D(fake_images).mean().item()))

    if ep + 1 == 1:
        out = images
        out = image_range(out)
        save_image(out, os.path.join(result_path, 'real_img.png'))

    out = fake_images
    out = image_range(out)
    save_image(out, os.path.join(result_path, 'fake_img {}.png'.format(ep)))

    tc.save(G.state_dict(), 'G.ckpt')
    tc.save(D.state_dict(), 'D.ckpt')