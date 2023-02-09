import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import numpy as np

# Get cpu or gpu device for training.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


training_data = datasets.CIFAR10(root="./data", download=True,
                                      train=True, 
                                      transform=transform_train)

test_data = datasets.CIFAR10(root="./data", download=True,
                                  train=False,
                                  transform=transform_test)

#augmented data
transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.RandomInvert(0.2),
        transforms.RandomCrop((26,26)),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

more_training_data = datasets.CIFAR10(root="./data", download=True,
                                      train=True, 
                                      transform=transformations)

augmented_training_data = torch.utils.data.ConcatDataset([training_data, more_training_data])

batch_size = 32

train_dataloader = DataLoader(augmented_training_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## to see accuracy during the training (and don't need to wait the end)
# writer = SummaryWriter(log_dir='my_logs')

for X, y in test_dataloader:
    X_shape = X.shape
    y_shape = y.shape
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# function to plot one image
def plot_image(image):
    #we dont want it normalized
    img = image / 2 + 0.5

    np_img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_dataloader)
images, labels = next(dataiter)

# show images
## plot all image on one grid 
for i in range(len(images)):
    print("Label of the image number", i, "is", classes[labels[i]])

    ## Only to see one image per one and the type of transform we applied
    # tr = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(90),
    #     transforms.RandomInvert(0.2),
    #     transforms.RandomCrop((26,26)),
    #     transforms.Resize((32,32)),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # plot_image(tr(images[i]))
    # plot_image(images[i])

plot_image(torchvision.utils.make_grid(images))

## plot images one by one
#for i in range(len(images)):
#    print("Label of the image", i, "is", classes[labels[i]])
#    plot_image(images[i])

num_classes = 10

class NeuralNetwork(nn.Module):
    #####################################
    ##       Convolutionnal Network   ##
    ####################################
    def __init__(self):
        super().__init__()
        ####
        ## RESNET
        ###

        self.model_ft = models.resnet18(pretrained=True)
        self.model_ft.fc = nn.Identity()
        self.l1 = nn.Linear(512, 256)
        self.b1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(0.2)
        self.l2 = nn.Linear(256, 128)
        self.b2 = nn.BatchNorm1d(128)
        self.d2 = nn.Dropout(0.2)
        self.l3 = nn.Linear(128, 64)
        self.b3 = nn.BatchNorm1d(64)
        self.d3 = nn.Dropout(0.2)
        self.l4 = nn.Linear(64, num_classes)

        ### 
        ## OUR NETWORK 
        ###

        # self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding='same')
        # self.batch1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding='same')
        # self.batch2 = nn.BatchNorm2d(128)
        # self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding='same')
        # self.batch3 = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding='same')
        # self.batch4 = nn.BatchNorm2d(512)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(2*2*512 , 128)
        # self.batch5 = nn.BatchNorm1d(128)
        # self.fc2 = nn.Linear(128, 64)
        # self.batch6 = nn.BatchNorm1d(64)
        # self.dropout = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(64, num_classes)
        # self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.model_ft(x).to(device)
        x = self.l1(x)
        x = self.sig(self.b1(x))
        x = self.d1(x)
        x = self.l2(x)
        x = self.sig(self.b2(x))
        x = self.d2(x)
        x = self.l3(x)
        x = self.sig(self.b3(x))
        x = self.d3(x)
        x = self.l4(self.sig(x))

        # x = self.conv1(x)
        # x = self.pool(self.relu(self.batch1(x)))
        # x = self.conv2(x)
        # x = self.pool(self.relu(self.batch2(x)))
        # x = self.dropout(x)
        # x = self.conv3(x)
        # x = self.pool(self.relu(self.batch3(x)))
        # x = self.conv4(x)
        # x = self.pool(self.relu(self.batch4(x)))
        # x = self.dropout(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = self.relu(self.batch5(x))
        # x = self.fc2(x)
        # x = self.relu(self.batch6(x))
        # x = self.dropout(x)
        # x = self.fc3(x)
        x = self.tanh(x)
        #x = self.soft(x)
        
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

    loss = running_loss / len(train_dataloader)
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

loss_points = torch.zeros(epochs)
acc_points = torch.zeros(epochs)

## Evaluate
y_pred = []
y_truth = []

running_loss = 0
running_acc = 0
print("Evaluating model ...")
with torch.no_grad():
    for batch, (X, y) in enumerate(test_dataloader):

        ## one-hot encoding our y data
        y = nn.functional.one_hot(y, 10).to(torch.float32)    
        X, y = X.to(device), y.to(device)

        #forward pass & loss
        pred = model(X)
        for p in pred:
            y_pred.append(p.cpu())
        
        for r in y:
            y_truth.append(r.cpu())

        total += y.size(0)
        running_loss += loss_fn(pred, y).item()

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

print("Loss during test :", running_loss/len(test_dataloader))
print("Accuracy of the model on test set : ", (100 * running_acc) / total, "%\n")

####
## Finding 10 worst classified images 
####

i = 0
dist_tab = []
real_labels = []

for X, y in test_dataloader:
    m = 0
    ind = 0
    for j in range(10):
        dist = torch.sum(y[j] - y_pred[i][j]).item()
        if (m < dist):
            m = dist
            ind = j
    dist_tab.append(m)
    real_labels.append(ind)
    i = i + 1

worst_images_ind = []
for i in range(10):
    m = dist_tab[0]
    index = 0
    for d in range(len(dist_tab)):
        if (m < dist_tab[d]):
            m = dist_tab[d]
            index = d
    worst_images_ind.append(index)
    dist_tab.remove(m)

for idx in worst_images_ind:
    ind_pred = 0
    m_prob = torch.tensor([0])
    for i in range(len(y_pred[idx])):
        if (m_prob[0] < y_pred[idx][i]):
            ind_pred = i
            m_prob[0] = y_pred[idx][i]
    print("Image n°=", idx)
    print("  > Classified as :", classes[ind_pred], "with a probability of", torch.max(y_pred[idx]).item()*100, "%")
    print("  > In reality it's a", classes[real_labels[idx]],"\n")
    ind_pred += 1

###
## Confusion Matrix
###

#one-hot encoding 
for i in range(len(y_pred)):
    index_max = torch.argmax(y_pred[i])
    for j in range(10):
        if (j == index_max):
            y_pred[i][j] = 1
        else:
            y_pred[i][j] = 0

y_truth_names = []
y_pred_names = []

for i in range(len(y_pred)):
    for j in range(10):
        if (y_pred[i][j] == 1):
            y_pred_names.append(classes[j])

for i in range(len(y_truth)):
    for j in range(10):
        if (y_truth[i][j] == 1):
            y_truth_names.append(classes[j])

confusion_matrix = metrics.confusion_matrix(y_truth_names, y_pred_names)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix)
cm_display.plot()
plt.show()




#First accuracy : 82.2 %
# Try first to add a batchNorm layer after each conv2d layer 
# Then maybe add conv2d layer and/or more filters ? (We had first one more conv layer (we got only 2 at the beginning) and we add more filters for each one)
# With only 10 epochs of training and with batchnorm2D layers we get around 87% accuracy
# We stay with 10 epochs for training, we have 3 layers convolution and have more filters (from 64 to 128 until 256 for last one) : 94% accuracy 
# 3rd Optimisation : Add dropout layers and multiply epochs by 2 : 95.4 % accuracy 
# 4rth (Last ?) : add a last conv2d layer can be maybe better and reorganize dropout and conv : we got between 96% and 97%  

## For DATA AUGMENTATION in the report : show pictures before and after transformations
## Data Augmentation ( plots ), we got only around 85 % accuracy but it's okay because we only want to reduce overfitting (see the section below)
## Improvment : adding Softmax layer in last layer we got 85.81 % accuracy and loss of 1.9 in testing (to see curves + matrix)
## Softmax argiment equal to dimension of output (example 2 : if we got (x, y, z, a) we will have (x*y*z, a))
## With tanh : Accuracy around 85% (85.02 %) loss around 1.7 in testing
## Worst images classified are (see screen)

## RESNET 
## We only add a few layers instead of the end layer of this network

