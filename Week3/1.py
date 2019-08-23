import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

bs = 100

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
# 第一次需要下载 download = True
train_dataset = datasets.MNIST(root='./MNIST_data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./MNIST_data/', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


z_dim = 100
# train_dataset.train_data.size (60000, 28, 28)
mnist_dim = train_dataset.train_data.size(1) * train_dataset.train_data.size(2)
G = Generator(input_dim=z_dim, output_dim=mnist_dim).to(device)
D = Discriminator(mnist_dim).to(device)
# print(G)
# print(D)

criterion = nn.BCELoss()

lr = 0.0002
G_optimizer = optim.Adam(G.parameters(), lr=lr)
D_optimizer = optim.Adam(D.parameters(), lr=lr)


def d_train(x):
    D.zero_grad()
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    d_output = D(x_real)
    d_real_loss = criterion(d_output, y_real)

    z = torch.randn(bs, z_dim).to(device)
    x_fake, y_fake = G(z), torch.zeros(bs, 1).to(device)

    d_output = D(x_fake)
    d_fake_loss = criterion(d_output, y_fake)

    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    D_optimizer.step()

    return d_loss.data.item()


def g_train(x):
    G.zero_grad()
    z = torch.randn(bs, z_dim).to(device)
    y = torch.ones(bs, 1).to(device)

    g_output = G(z)
    d_output = D(g_output)
    g_loss = criterion(d_output, y)

    g_loss.backward()
    G_optimizer.step()

    return g_loss.data.item()


n_epoch = 200

for epoch in range(1, n_epoch + 1):
    D_losses, G_losses = [], []
    for batch_idx, (img, label) in enumerate(train_loader):
        D_losses.append(d_train(img))
        G_losses.append(g_train(img))
    print('[', epoch, '/', n_epoch, ']:', torch.mean(torch.FloatTensor(D_losses)),
          torch.mean((torch.FloatTensor(G_losses))))

with torch.no_grad():
    test_z = torch.randn(bs, z_dim)
    g = G(test_z)
    save_image(g.view(g.size(0), 1, 28, 28), '/samples/sample_' + '.png')
