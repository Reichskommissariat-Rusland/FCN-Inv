import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


# Step 1: Load CIFAR10 data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root='./dataset/cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=2)

# Step 2: Define the neural network


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(3*32*32, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


net = Net()

# Step 3: Define a Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Step 4: Train the network on the training data
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs_clone = inputs.clone()  # Clone the inputs tensor

    # De-normalize and save the image before feeding to the network
    img = inputs_clone.squeeze().detach().numpy()  # Convert to numpy array
    print(img.shape)
    # Transpose to (height, width, channels)
    img = np.transpose(img, (1, 2, 0))
    img = (img * 0.5 + 0.5) * 255  # De-normalize to [0, 255]
    img = img.astype(np.uint8)  # Convert to uint8
    img_pil = Image.fromarray(img)  # Convert to PIL Image
    img_pil.save(f'input_image_{i}.png')  # Save the image

    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()

    optimizer.step()

    # Stop after the first image
    break

grad_b = net.fc1.bias.grad.numpy().reshape((120, 1))
grad_w = net.fc1.weight.grad.numpy()

inversion_res = np.matmul(np.linalg.pinv(grad_b), grad_w).reshape((3, 32, 32))
img = np.transpose(inversion_res, (1, 2, 0))
img = (img * 0.5 + 0.5) * 255  # De-normalize to [0, 255]
img = img.astype(np.uint8)  # Convert to uint8
img_pil = Image.fromarray(img)  # Convert to PIL Image
img_pil.save(f'reverse_image_{i}.png')  # Save the image
