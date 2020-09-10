import torch
from torch import nn
from torch.nn import functional as f
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Constantes
POOL_SIZE = 1024
N_CHANNELS = 16
width = height = int(math.sqrt(POOL_SIZE))
# Filtros

sobelX = torch.from_numpy(np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(float)).repeat((16, 16, 1, 1))
sobelY = torch.from_numpy(np.array(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(float)).repeat((16, 16, 1, 1))
cellId = torch.from_numpy(np.array(
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(float)).repeat((16, 16, 1, 1))

filters = [sobelX, sobelY, cellId]


grid = np.zeros((width, height, N_CHANNELS))


class UpdateGrid(nn.Module):

    def __init__(self):
        super(UpdateGrid, self).__init__()

        # Se usan Conv2D ya que es una forma de aplicar una capa Dense en todo
        # el tablero
        self.fc1 = nn.Conv2d(N_CHANNELS * len(filters), 128, (1, 1))
        self.fc2 = nn.Conv2d(128, N_CHANNELS, (1, 1))

    def forward(self, x):

        # Cada pixel tiene 16 canales.
        # Por eso tiene las dimensiones de ancho y alto de la imagen.
        # Los 16 canales son para cada filtro.
        perception = torch.empty((1, len(filters) * N_CHANNELS, width, height))

        for n, filt in enumerate(filters):
            perception[:, n * N_CHANNELS:
                       (n + 1) * N_CHANNELS, :, :] = f.conv2d(x, filt, padding=[1, 1])

        dx = self.fc1(perception)
        dx = f.relu(dx)
        dx = self.fc2(dx)

        # Random Mask
        randomMask = torch.from_numpy(np.random.randint(
            0, 2, (1, 1, width, height))).repeat((1, 16, 1, 1))
        # Skip conecction + stochastic update
        x = x + dx * randomMask

        # Aplicando Filtro para poder saber el estado de las celulas vecinas
        alive = f.conv2d((x[:, 3:4, :, :] > 0.1).type(torch.int), torch.from_numpy(
            np.ones((1, 1, 3, 3)).astype(int)), padding=1)

        alive = (alive > 0).type(torch.int)

        return x * alive.repeat(1, 16, 1, 1)


im = Image.open('wow.png')

grid[:, :, 0:4] = np.array(im) / 255

# NN
updateGrid = UpdateGrid()

result = updateGrid.forward(torch.from_numpy(
    grid).view((1, width, height, N_CHANNELS)).permute(0, 3, 1, 2))

for step in range(500):

    if step == 0:
        result = updateGrid.forward(torch.from_numpy(
            grid).view((1, width, height, N_CHANNELS)).permute(0, 3, 1, 2))
    else:
        result = updateGrid.forward(result)

    imgRes = result[0, 0:4, :, :].detach().numpy().transpose(1, 2, 0)

    plt.imshow(imgRes)
    plt.show()
