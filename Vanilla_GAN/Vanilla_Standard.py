import torch as tc
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
import os
from collections import namedtuple

#TODO: Tunable noise_sz and hidden_sz.
noise_sz = 100
hidden_sz = 512

#MNIST: 0, FashionMNIST: 1, CIFAR10: 2

def method_check(method, batch_sz, trainloader, parameter_tuple, total_step):
    if method == 0:
        result_dir = './MNIST_results'
        pt_name = ('MNIST_G.ckpt', 'MNIST_D.ckpt')
        channel = 1
        height = 28
        width = 28
        img_sz = channel * height * width

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.MNIST(root='./MNIST_data', train=True, download=True, transform=transform)
        trainloader.append(tc.utils.data.DataLoader(dataset=trainset, batch_size=batch_sz, shuffle=True))

    elif method == 1:
        result_dir = './Fashion_results'
        pt_name = ('Fashion_G.ckpt', 'Fashion_D.ckpt')
        channel = 1
        height = 28
        width = 28
        img_sz = channel * height * width

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.FashionMNIST(root='./FashionMNIST_data', train=True, download=True,
                                                     transform=transform)
        trainloader.append(tc.utils.data.DataLoader(dataset=trainset, batch_size=batch_sz, shuffle=True))

    elif method == 2:
        result_dir = './CIFAR10_results'
        pt_name = ('CIFAR10_G.ckpt', 'CIFAR10_D.ckpt')
        channel = 3
        height = 32
        width = 32
        img_sz = channel * height * width

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./CIFAR10_data', train=True, download=True, transform=transform)
        trainloader.append(tc.utils.data.DataLoader(dataset=trainset, batch_size=batch_sz, shuffle=True))

    # the size of trainset
    total_step.append(len(trainloader[0]))
    print("total step : ", total_step[0])

    parameter = namedtuple("parameter", "channel, height, width, img_sz, result_dir, pt_name")
    parameter_tuple.append(parameter(channel, height, width, img_sz, result_dir, pt_name))

    return True

def network_optimizer(img_sz, lr):
    # Generator
    G = nn.Sequential(
        nn.Linear(noise_sz, hidden_sz),
        nn.ReLU(),
        nn.Linear(hidden_sz, hidden_sz),
        nn.ReLU(),
        nn.Linear(hidden_sz, img_sz),
        nn.Tanh()  # [-1, 1], Image value)
    )

    # Discriminator
    D = nn.Sequential(
        nn.Linear(img_sz, hidden_sz),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.5),
        nn.Linear(hidden_sz, hidden_sz),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.5),
        nn.Linear(hidden_sz, 1),
        nn.Sigmoid()  # [0, 1], Fake or Real
    )

    # cuda or cpu
    gpu_checker = 'cuda' if tc.cuda.is_available() else 'cpu'
    print("You are", gpu_checker, "is available")
    device = tc.device(gpu_checker)

    G = G.to(device)
    D = D.to(device)

    criterion = nn.BCELoss()
    d_opt = tc.optim.Adam(D.parameters(), lr=lr)
    g_opt = tc.optim.Adam(G.parameters(), lr=lr)

    return G, D, device, criterion, d_opt, g_opt

def Img_Range(x):
    out = (x+1)/2
    out = out.clamp(0,1)
    return out

def train(method, batch_sz, epoch_sz, lr, log_wirter):
    trainloader=[]
    parameter_tuple=[]
    total_step=[]
    if not method_check(method, batch_sz, trainloader, parameter_tuple, total_step) :
        return
    # parameter setting
    channel = parameter_tuple[0].channel
    height = parameter_tuple[0].height
    width = parameter_tuple[0].width
    img_sz = parameter_tuple[0].img_sz
    result_dir = parameter_tuple[0].result_dir
    pt_name = parameter_tuple[0].pt_name
    G, D, device, criterion, d_opt, g_opt = network_optimizer(img_sz, lr)
    G.train(True)
    D.train(True)

    # result path
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)


    for epoch in range(epoch_sz):
        for batch_idx, (image, _) in enumerate(trainloader[0]):
            mini = image.size()[0]
            image = image.reshape(mini, -1).to(device)  # mini batch

            real_label = tc.ones(mini, 1).to(device)
            fake_lable = tc.zeros(mini, 1).to(device)

            ####Train the Discriminator###
            ###for real###
            output = D(image)
            d_loss_real = criterion(output, real_label)
            real_score = output

            ###for fake###
            z = tc.randn(batch_sz, noise_sz).to(device)
            fake_image = G(z)
            output = D(fake_image)
            d_loss_fake = criterion(output, fake_lable)
            fake_score = output

            d_loss_total = d_loss_real + d_loss_fake

            D.zero_grad()
            d_loss_total.backward()
            d_opt.step()

            ####Train the Generator####
            z = tc.randn(batch_sz, noise_sz).to(device)
            fake_image = G(z)
            output = D(fake_image)

            g_loss = criterion(output, real_label)

            G.zero_grad()
            g_loss.backward()
            g_opt.step()
            # c.f item() : to get python number (tensor(xx) is not represented)
            if batch_idx % 200 == 0:
                log_wirter.writerow(['{}/{}'.format(epoch, epoch_sz),
                                '{}/{}'.format(batch_idx + 1, total_step[0]),
                                d_loss_total.item(),
                                g_loss.item(),
                                real_score.mean().item(),
                                fake_score.mean().item()])
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch, epoch_sz, batch_idx + 1, total_step[0], d_loss_total.item(), g_loss.item(),
                              real_score.mean().item(), fake_score.mean().item()))

        # Save the real data for comparision
        if epoch == 0:
            image = image.reshape(image.size(0), channel, height, width)
            save_image(Img_Range(image), os.path.join(result_dir, 'real_image.png'))

        fake_image = fake_image.reshape(fake_image.size(0), channel, height, width)
        save_image(Img_Range(fake_image), os.path.join(result_dir, 'fake_image{}.png'.format(epoch)))

        # Save the check point
        tc.save(G.state_dict(), pt_name[0])
        tc.save(D.state_dict(), pt_name[1])

    print("Processing done")

#for test
#def test():
    # pt = tc.load('G.ckpt')
    # G.load_state_dict(pt)
    # z = tc.randn(noise_sz).to(device)
    # fake_image = G(z)
    # fake_image = fake_image.reshape(channel, height, width)
    # fake_image = (fake_image+1)/2
    # fake_image = fake_image.clamp(0,1)
    # save_image(fake_image, os.path.join(result_dir, 'fake_image_test.png'))