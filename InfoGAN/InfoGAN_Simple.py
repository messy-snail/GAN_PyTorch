import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import numpy as np

#TODO : Test 해보기. 발산함.
#epoch_sz = 100
epoch_sz = 300
#img_sz = 64
latent_sz = 100
batch_sz = 128
lr = 0.0002
nChannel = 1
dc_sz = 10
cc_sz = 1

#for mlp
hidde_sz = 1024
img_sz = 784

lambda_dc = 1
lambda_cc = 0.1 #smaller lamda

result_path = 'result'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

#trans = transforms.Compose([transforms.Resize((img_sz, img_sz)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#for MLP
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = tv.datasets.MNIST(root='./MNIST_data', train=True, transform=trans, download=True)
dataloader = tc.utils.data.DataLoader(dataset=dataset, batch_size=batch_sz, shuffle= True)

def img_range(x):
    out = (x+1)/2
    out = out.clamp(0,1)
    return out

def normal_init(m, mean, std):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# class Generator(nn.Module):
#     def __init__(self, latent_sz):
#         super(Generator, self).__init__()
#         self.tconv1 = nn.ConvTranspose2d(latent_sz, 1024, 4, 1, 0)
#         self.tconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
#         self.tconv3 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
#         self.tconv4 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
#         self.tconv5 = nn.ConvTranspose2d(128, nChannel, 4, 2, 1)
#
#         self.bn1 = nn.BatchNorm2d(1024)
#         self.bn2 = nn.BatchNorm2d(512)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.bn4 = nn.BatchNorm2d(128)
#
#     def forward(self, latent):
#         x = F.relu(self.bn1(self.tconv1(latent)))
#         x = F.relu(self.bn2(self.tconv2(x)))
#         x = F.relu(self.bn3(self.tconv3(x)))
#         x = F.relu(self.bn4(self.tconv4(x)))
#         x = F.tanh(self.tconv5(x))
#
#         return x
#
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.conv1 = nn.Conv2d(nChannel, 128, 4, 2, 1)
#         self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
#         self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
#         self.conv4 = nn.Conv2d(512, 1024, 4, 2, 1)
#         #self.conv5 = nn.Conv2d(1024, 1+cat_sz+con_sz, 4, 1, 0)
#         self.conv5 = nn.Conv2d(1024, 1 + dc_sz + cc_sz, 4, 1, 0)
#
#         self.bn2 = nn.BatchNorm2d(256)
#         self.bn3 = nn.BatchNorm2d(512)
#         self.bn4 = nn.BatchNorm2d(1024)
#
#     def forward(self, images):
#         x = F.leaky_relu(self.conv1(images), 0.2)
#         x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
#         x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
#         x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
#         x = F.sigmoid(self.conv5(x))
#
#         return x
#
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)
#
# G = Generator(latent_sz+dc_sz + cc_sz)
# D = Discriminator()
#
# G.weight_init(mean=0.0, std=0.02)
# D.weight_init(mean=0.0, std=0.02)

G = nn.Sequential(
    nn.Linear(latent_sz+dc_sz + cc_sz, hidde_sz),
    #nn.BatchNorm1d(hidde_sz),
    nn.ReLU(),
    nn.Linear(hidde_sz, hidde_sz),
    #nn.BatchNorm1d(hidde_sz),
    nn.ReLU(),
    nn.Linear(hidde_sz, hidde_sz),
    #nn.BatchNorm1d(hidde_sz),
    nn.ReLU(),
    nn.Linear(hidde_sz, img_sz),
    nn.Tanh()
)

D = nn.Sequential(
    nn.Linear(img_sz, hidde_sz),
    nn.LeakyReLU(0.2),
    nn.Linear(hidde_sz, hidde_sz),
    #nn.BatchNorm1d(hidde_sz),
    nn.LeakyReLU(0.2),
    nn.Linear(hidde_sz, hidde_sz),
    #nn.BatchNorm1d(hidde_sz),
    nn.LeakyReLU(0.2),
    nn.Linear(hidde_sz, 1+dc_sz+cc_sz),
    nn.Sigmoid()
)

device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')
G.to(device)
D.to(device)

loss_func = nn.BCELoss()
dis_loss_func = nn.CrossEntropyLoss()

g_opt = tc.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
d_opt = tc.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


for ep in range(epoch_sz):
    for step, (images, _) in enumerate(dataloader):
        images = images.to(device)
        mini_batch = images.size()[0]

        #for MLP
        images = images.reshape(mini_batch, -1)

        #noise =tc.randn(mini_batch, latent_sz).view(-1, latent_sz, 1, 1).to(device)
        noise = tc.randn(mini_batch, latent_sz).to(device)

        temp_c1 = np.random.randint(0, dc_sz, mini_batch)
        c1 = tc.zeros(mini_batch, dc_sz)
        c1[range(mini_batch), temp_c1] =1

        #c1 = tc.randint(0, 10, (mini_batch,dc_sz))*0.1
        c2 = tc.randn((mini_batch,cc_sz))

        c1=c1.to(device)
        c2=c2.to(device)

        #for MLP
        z = tc.cat([noise, c1, c2], 1).view(-1, latent_sz + dc_sz + cc_sz).to(device)

        #z = tc.cat([noise, c1, c2], 1).view(-1, latent_sz + dc_sz + cc_sz, 1, 1).to(device)

        real_label = tc.ones(mini_batch).to(device)
        fake_label = tc.zeros(mini_batch).to(device)

        # D_real = D(images).squeeze()
        # D_fake = D(G(z)).squeeze()

        #for MLP
        D_real = D(images)
        D_fake = D(G(z))

        #D_real = D(images)
        #D_fake = D(G(z))

        # loss_real = loss_func(D_real, real_label)
        # loss_fake = loss_func(D_fake, fake_label)

        #noise
        d_loss = -tc.mean(tc.log(D_real[:,0]) + tc.log(1-D_fake[:,0]))

        # noise, dc ,cc
        out_dc = D_fake[:, 1:1 + dc_sz]
        out_cc = D_fake[:, 1 + dc_sz:]

        #TODO:loss analysis
        d_loss_cc = tc.mean((((out_cc - 0.0) / 0.5) ** 2))
        d_loss_dc = -(tc.mean(tc.sum(c1 * out_dc, 1)) + tc.mean(tc.sum(c1 * c1, 1)))

        d_loss_total = d_loss+d_loss_dc*lambda_dc+d_loss_cc*lambda_cc

        #TODO : retain_graph
        D.zero_grad()
        d_loss_total.backward(retain_graph=True)
        d_opt.step()

        # noise = tc.randn(mini_batch, latent_sz).to(device)
        # c1 = tc.randint(0, dc_sz, (mini_batch,dc_sz))*0.1
        # c2 = tc.randn((mini_batch,cc_sz))
        #
        # c1 = c1.to(device)
        # c2 = c2.to(device)
        #
        # #z = tc.cat([noise, c1, c2], 1).view(-1, latent_sz+con_sz+cat_sz, 1, 1).to(device)
        # z = tc.cat([noise, c1, c2], 1).view(-1, latent_sz +  dc_sz + cc_sz, 1, 1).to(device)

        fake_imgs = G(z)
        D_result = D(fake_imgs).squeeze()
        #g_loss = loss_func(D_result, real_label)

        #TODO:loss analysis
        g_loss = -tc.mean(tc.log(D_result[:,0]))
        g_loss_total = g_loss+d_loss_dc*lambda_dc+d_loss_cc*lambda_cc

        G.zero_grad()
        g_loss_total.backward()
        g_opt.step()

        if step%200 ==0:
            print('epoch {}/{}, step {}, d_loss {:.4f}, g_loss {:.4f}, Real_score {:.2f}, Fake_score {:.2f}'.format(ep, epoch_sz, step, d_loss.item(), g_loss.item(), D(images).mean().item(), D(fake_imgs).mean().item()))

    if ep+1==1:
        out = images
        #for MLP
        out = out.reshape(mini_batch, 1, 28, 28)

        out = img_range(out)
        save_image(out, os.path.join(result_path, 'real_img.png'))

    out = fake_imgs
    #for MLP
    out = out.reshape(mini_batch, 1, 28, 28)

    out = img_range(out)
    save_image(out, os.path.join(result_path, 'fake_img{}.png'.format(ep)))

    tc.save(G.state_dict(), 'G.ckpt')
    tc.save(D.state_dict(), 'D.ckpt')


##########TEST MODULE##########

# test_path = 'result/test'
# if not os.path.isdir(test_path):
#     os.mkdir(test_path)
#
#
# pt = tc.load('G.ckpt')
# G.load_state_dict(pt)
# #for MLP
# # for ep in range(epoch_sz):
# for  i in range(10):
#     noise = tc.randn(16, latent_sz).to(device)
#     c1 = tc.ones(16, 10) * 0.1 * i
#     # c1 = tc.randint(0, dc_sz, (100,dc_sz))*0.1
#     c2 = tc.randn((16, cc_sz))
#
#     c1 = c1.to(device)
#     c2 = c2.to(device)
#
#     z = tc.cat([noise, c1, c2], 1).view(-1, latent_sz + dc_sz + cc_sz).to(device)
#     fake_image = G(z)
#     fake_image = fake_image.reshape(-1, 1, 28, 28)
#     fake_image = (fake_image+1)/2
#     fake_image = fake_image.clamp(0,1)
#     save_image(fake_image, os.path.join(test_path, 'fake_image_test{}.png'.format(i)))
