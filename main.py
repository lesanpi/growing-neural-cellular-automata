import torch
from torch import nn
from torch.nn import functional as f
import torch.optim as optim
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
        torch.nn.init.zeros_(self.fc2.weight)

    def forward(self, x):

        # Cada pixel tiene 16 canales.
        # Por eso tiene las dimensiones de ancho y alto de la imagen.
        # Los 16 canales son para cada filtro.
        perception = torch.empty((1, len(filters) * N_CHANNELS, width, height))

        # Filtros
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

        alive = (alive > 0.0).type(torch.int)

        return x * alive.repeat(1, 16, 1, 1)


im = Image.open('wow.png')
# Imagen objetivo.
target = torch.tensor(np.array(im) / 255)

# NN
updateGrid = UpdateGrid()
# Loss Function
loss_f = nn.MSELoss()
# Creando el optimizador.
optimizer = optim.Adam(updateGrid.parameters(), lr=1e-5)

# Training loop
for trStep in range(1000):

    # Numero de pasos aleatorios.
    n_steps = np.random.randint(64, 96)
    debug = n_steps - 1

    # Inicializar la grilla con una sola celula en el centro.
    grid = np.zeros((width, height, N_CHANNELS))
    grid[height // 2, width // 2, 3:] = 1
    # La imagen sin pasar por la NN
    result = torch.from_numpy(
        grid).view((1, width, height, N_CHANNELS)).permute(0, 3, 1, 2)

    for step in range(n_steps):

        result = torch.clamp(updateGrid.forward(result), 0.0, 1.0)

        # Muestra el estado cada n veces.
        if step + 1 == n_steps:

            imgRes = np.clip(result[0, 0:4, :, :].detach(
            ).numpy().transpose(1, 2, 0)[:, :, :4], 0.0, 1.0)
            plt.imshow(imgRes)
            plt.show()

    output = result[0, 0:4, :, :].transpose(0, 2)
    # Entrenamiento.
    optimizer.zero_grad()
    loss = loss_f(output, target)
    loss.backward()
    optimizer.step()

    # Print
    print("Tr. Step: " + str(trStep) + " Training loss: " + str(loss))
