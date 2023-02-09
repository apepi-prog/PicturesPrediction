import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Get cpu or gpu device for training.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

training_data = datasets.MNIST(root="./data", download=True,
                                      train=True, 
                                      transform=transforms.ToTensor())

test_data = datasets.MNIST(root="./data", download=True,
                                  train=False,
                                  transform=transforms.ToTensor())
# data is in [0; 1] (thanks to ToTensor()), but there is no "standardisation"


batch_size = 32
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

## to see accuracy during the training (and don't need to wait the end)
#writer = SummaryWriter(log_dir='my_logs')

for X, y in test_dataloader:
    X_shape = X.shape
    y_shape = y.shape
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

input_size = X.shape[2] * X.shape[3]
num_classes = 10

class NeuralNetwork(nn.Module):
    #####################################
    ##       Convolutionnal Network   ##
    ####################################
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding='same')
        self.batch1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding='same')
        self.batch2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(7*7*128, 128)
        self.batch3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.batch4 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.relu(self.batch1(x)))
        x = self.conv2(x)
        x = self.pool(self.relu(self.batch2(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(self.batch3(x))
        x = self.fc2(x)
        x = self.relu(self.batch4(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

model = NeuralNetwork().to(device)
print(model)

lr = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

# # Train
epochs = 10

#arrays to get our plots at the end
x_points = torch.arange(epochs)
loss_points = torch.zeros(epochs)
acc_points = torch.zeros(epochs)

for e in range(epochs):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    
    running_acc = 0
    running_loss = 0
    total = 0
    for batch, (X, y) in enumerate(train_dataloader):
        
        ## one-hot encoding our y data
        y = nn.functional.one_hot(y, 10).to(torch.float32)
        
        X, y = X.to(device), y.to(device)

        #forward pass & loss
        pred = model(X)
        loss = loss_fn(pred, y)
        total += y.size(0)

        #backward pass / backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        #computing accuracy
        y_hat = torch.zeros(len(y), 10)
        
        for i in range(len(y)):
            for j in range(10):
                if (pred[i][j] >= 0.5):
                    y_hat[i][j] = 1
        
        for i in range(len(y)):
            for j in range(10):
                if (y[i][j] == y_hat[i][j]):
                    running_acc += 1

                    

    loss = running_loss / len(test_dataloader)
    accuracy = 100 * running_acc / (total*10)

    loss_points[e] = loss
    acc_points[e] = accuracy

    print(f'{e}/{epochs}: loss={loss:.4f} acc={accuracy:.4f}')

    # To see the evolution of accuracy and loss during training 
    #writer.add_scalar('Loss/train', loss, e)
    #writer.add_scalar('Acc/train', accuracy, e)
print("\n")

### display our plots 
plt.plot(x_points, loss_points)
plt.xlabel("Epochs")
plt.ylabel("Loss during training")
plt.show()

plt.plot(x_points, acc_points)
plt.xlabel("Epochs")
plt.ylabel("Accuracy during training")
plt.show()

## Evaluate
print("EVALUATION OF THE MODEL \n")
for batch, (X, y) in enumerate(test_dataloader):

    ## one-hot encoding our y data
    y = nn.functional.one_hot(y, 10).to(torch.float32)
        
    X, y = X.to(device), y.to(device)

    #forward pass & loss
    pred = model(X)
    total += y.size(0)

    pred = torch.where(pred > 0.5, 1, 0)

    #computing accuracy
    y_hat = torch.zeros(len(y), 10)
        
    for i in range(len(y)):
        for j in range(10):
            if (pred[i][j] >= 0.5):
                y_hat[i][j] = 1
        
    for i in range(len(y)):
        for j in range(10):
            if (y[i][j] == y_hat[i][j]):
                running_acc += 1
print("Accuracy of the model on test set : ", (100 * running_acc / (total*10)), "%\n")

# We got 94.335 % of accuracy on the test set (First model 2 conv Layers + maxPool )
# We got a loss of 0.0057 at the end of training 
# To get better accuracy we add batchnorm layers and dropout layers and add more filters to conv layers (from 6 & 64 to 128 & 248) : 99.1 % accuracy 

### TO GET CONFUSION MATRIX SEE :
# https://torchmetrics.readthedocs.io/en/stable/classification/confusion_matrix.html
# https://pytorch.org/ignite/generated/ignite.metrics.confusion_matrix.ConfusionMatrix.html
# https://datascientest.com/matrice-de-confusion
