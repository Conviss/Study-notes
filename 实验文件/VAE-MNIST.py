import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



def loss_function(recon_x, x, mu, logvar):
    """
    :param recon_x: generated image
    :param x: original image
    :param mu: latent mean of z
    :param logvar: latent log variance of z
    """
    BCE_loss = nn.BCELoss(reduction='sum')
    reconstruction_loss = BCE_loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1+logvar-torch.exp(logvar)-mu**2)
    #KLD_ele = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    #KLD = torch.sum(KLD_ele).mul_(-0.5)
    # print(reconstruction_loss, KL_divergence)

    return reconstruction_loss + KL_divergence


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2_mean = nn.Linear(400, 20)
        self.fc2_logvar = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparametrization(self, mu, logvar):
        # sigma = exp(0.5*log(sigma^2))= exp(log(sigma))
        std = torch.exp(0.5 * logvar)
        # N(mu, std^2) = N(0, 1) * std + mu

        # 从标准正态分布中选择于std.size()相同大小的值
        z = torch.randn(std.size()) * std + mu
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        # 算出 均值和log方差
        mu, logvar = self.encode(x)
        # 重新采样， 因为直接对Z采用运算是不可导的，所以利用对正态分布采样在运算使其可导，Z = N~(0,1) * std + mu
        z = self.reparametrization(mu, logvar)
        # 返回对z重新编码
        return self.decode(z), mu, logvar


# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5]),
# ])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

vae = VAE()
optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003)

# Training
def train(epoch):
    # vae.train()
    all_loss = 0.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        # flatten(x,1)是按照x的第1个维度拼接（按照列来拼接，横向拼接）；
        # flatten(x,0)是按照x的第0个维度拼接（按照行来拼接，纵向拼接）；
        # 从start_dim开始到end_dim结束，其他不变，将其维度合在一起
        # print(inputs.size())
        real_imgs = torch.flatten(inputs, start_dim=1)
        # print(real_imgs.size())

        # Train Discriminator
        gen_imgs, mu, logvar = vae(real_imgs)
        loss = loss_function(gen_imgs, real_imgs, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss += loss.item()
        if batch_idx == 0: print('Epoch {}, loss: {:.6f}'.format(epoch, all_loss/(batch_idx+1)))



for epoch in range(20):
    train(epoch)

#创建画板并输出图像
fig, axes = plt.subplots(10, 2, figsize=(8, 8))
# ight_layout()可以接受关键字参数pad、w_pad或者h_pad，这些参数图像边界和子图之间的额外边距。边距以字体大小单位规定。

fig.tight_layout(pad=0.1,rect=[0, 0.03, 1, 0.92]) #[left, bottom, right, top] tight_layout会自动调整子图参数，使之填充整个图像区域。

for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to('cpu'), targets.to('cpu')
    real_imgs = torch.flatten(inputs, start_dim=1)
    gen_imgs, mu, logvar = vae(real_imgs)
    fake_images = gen_imgs.detach().view(-1, 1, 28, 28)

    for i, ax in enumerate(axes.flat):
        if i % 2 == 0:
            ax.imshow(inputs[i][0], cmap='gray', interpolation='none')  # 子显示
            ax.set_title("Ground Truth: {}".format(targets[i]))  # 显示title
        else:
            ax.imshow(fake_images[i - 1][0], cmap='gray', interpolation='none')  # 子显示
            ax.set_title("Ground Truth: {}".format(targets[i - 1]))  # 显示title
        ax.set_axis_off()

    plt.show()

    break

# torch.save(vae.state_dict(), './vae.pth')