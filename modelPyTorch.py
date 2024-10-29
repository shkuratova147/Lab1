import torch
import torch.nn as nn
# Создание нейронной сети
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)
# Инициализация модели
model = SimpleNN()
# Огляд структуры моделі
print(model)
