import torch
from torch import nn
from torch.nn import functional as f
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Constantes
POOL_SIZE = 1024
N_CHANNELS = 16
BATCH_SIZE = 8
width = height = int(math.sqrt(POOL_SIZE))
# Filtros

sobelX = torch.from_numpy(np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(float)).repeat((N_CHANNELS, 1, 1, 1))
sobelY = torch.from_numpy(np.array(
    [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(float)).repeat((N_CHANNELS, 1, 1, 1))
cellId = torch.from_numpy(np.array(
    [[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(float)).repeat((N_CHANNELS, 1, 1, 1))

filters = [sobelX, sobelY, cellId]


plt.ion()
plt.show()


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
        perception = torch.empty(
            (BATCH_SIZE, len(filters) * N_CHANNELS, width, height))

        # Filtros
        for n, filt in enumerate(filters):
            perception[:, n * N_CHANNELS:
                       (n + 1) * N_CHANNELS, :, :] = f.conv2d(x, filt, groups=N_CHANNELS, padding=[1, 1])

        dx = self.fc1(perception)
        dx = f.relu(dx)
        dx = self.fc2(dx)

        # Random Mask
        randomMask = torch.from_numpy(np.random.randint(
            0, 2, (BATCH_SIZE, 1, width, height))).repeat(1, 16, 1, 1)
        # Skip conecction + stochastic update
        x = x + dx * randomMask

        # Aplicando Filtro para poder saber el estado de las celulas vecinas
        alive_filter = torch.from_numpy(np.ones((1, 1, 3, 3)).astype(int))
        alive = f.conv2d((x[:, 3:4, :, :] > 0.1).double(), alive_filter.double(), padding=1)

        alive = (alive > 0.0)  # .type(torch.int)

        return x * alive.repeat(1, 16, 1, 1)


im = Image.open('luna.png')
# Imagen objetivo.
target = torch.tensor(np.array(im) / 255)

# NN
updateGrid = UpdateGrid()
# Loss Function
loss_f = nn.MSELoss()
# Creando el optimizador.
optimizer = optim.Adam(updateGrid.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=500, gamma=0.1)

for p in updateGrid.parameters():
    p.register_hook(lambda grad: grad / (torch.norm(grad, 2) + 1e-8))

debug = 10
n_epochs = 1000

# Training loop
for trStep in range(n_epochs + 1):

    # Numero de pasos aleatorios.
    n_steps = np.random.randint(64, 96)
    #n_steps - 1

    # Inicializar la grilla con una sola celula en el centro.
    grid = np.zeros((width, height, N_CHANNELS))
    grid[height // 2, width // 2, 3:] = 1.0

    batch_grid = np.repeat(grid[np.newaxis, ...], BATCH_SIZE, axis=0)

    # La imagen sin pasar por la NN
    result = torch.from_numpy(batch_grid).permute(0, 3, 1, 2)
    #result = torch.from_numpy(grid).view((1, width, height, N_CHANNELS)).permute(0, 3, 1, 2)

    for step in range(n_steps):

        result = torch.clamp(updateGrid.forward(result), 0.0, 1.0)

        # Muestra el estado cada n veces.
        if (trStep + 1) % debug == 0:

            batch_img = result[0, :4, :, :].detach().cpu().numpy()

            imgRes = np.clip(batch_img.transpose(1, 2, 0)[:, :, :4], 0.0, 1.0)
            plt.imshow(imgRes)

            plt.title('Tr.Step:' + str(trStep) + '- Step:' + str(step))
            plt.draw()
            plt.pause(0.01)
            plt.clf()
            # plt.show()

    # Limpiar los gradientes
    optimizer.zero_grad()

    result = torch.clamp(result, 0.0, 1.0)

    # RGBA
    output = result[:, :4, :, :].permute(0, 2, 3, 1)
    # Entrenamiento.

    loss = loss_f(output, target.repeat((BATCH_SIZE, 1, 1, 1)))
    loss.backward()
    # Optimizamos un paso
    optimizer.step()
    scheduler.step()

    # Print
    print('Tr.Loss ' + str(trStep) + ': ' + str(loss.item())
          [0:6] + '; lr=' + str(optimizer.param_groups[0]['lr']))
