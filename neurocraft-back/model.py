import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from flask import Flask, request, jsonify

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def dataset_loader(datas):
    train_dataset, test_dataset = {
        'mnist': (datasets.MNIST(root='./data', train=True, download=True, transform=transform), datasets.MNIST(root='./data', train=False, download=True, transform=transform)),
        'fashion_mnist': (datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform), datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)),
        'cifar10': (datasets.CIFAR10(root='./data', train=True, download=True, transform=transform), datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)),
        'cifar100': (datasets.CIFAR100(root='./data', train=True, download=True, transform=transform), datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)),
    }[datas]
    
    
    
    return (torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True),
            torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False) , 
            max(train_dataset.targets).item()+1, train_dataset[0][0].shape)


class NN(nn.Module):
    def __init__(self, input_size, num_classes, arch):
        super().__init__()
        self.device = device
        layerss = []
        sizes = []
        for l, v in arch:
            layerss.append(l)
            sizes.append(v)
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        # Input layer
        size = sizes[0]
        self.layers.append(nn.Linear(in_features=prev_size, out_features=size).to(self.device))
        
        # Hidden layers
        prev_size = size
        for i in range(1, len(arch)):
            size = sizes[i]
            layer = layerss[i]
            if layer == 'linear':
                self.layers.append(nn.Linear(in_features=prev_size, out_features=size).to(self.device))
            elif layer == 'relu':
                self.layers.append(nn.ReLU().to(self.device))
            elif layer == 'sigmoid':
                self.layers.append(nn.Sigmoid().to(self.device))
            elif layer == 'batchnorm1d':
                self.layers.append(nn.BatchNorm1d(prev_size).to(self.device))
            elif layer == 'dropout':
                self.layers.append(nn.Dropout().to(self.device))
            elif layer == 'flatten':
                self.layers.append(nn.Flatten().to(self.device))
            elif layer == 'softmax':
                self.layers.append(nn.Softmax(dim=1).to(self.device))
            if size is not None:
                prev_size = size
        
        # Output layer
        self.layers.append(nn.Linear(prev_size, num_classes).to(self.device))
        self.layers.append(nn.Softmax(dim=1).to(self.device))
  
    def forward(self, x):
        x = x.to(self.device)
        for layer in self.layers:
            x = layer(x)
        return x

@app.route('/train', methods=['POST'])
def run_train():
    data= request.get_json()
    dataset_name= data['dataset']
    arch= data['arch']
    iterations= data['iterations']
    
    
    (train_loader, test_loader), num_classes, shape = dataset_loader(dataset_name)
    
    for sampleimg, samplelbl in train_loader:
        break
    B, A, T, C = sampleimg.shape
    
    
    model = NN(input_size=(T*C), num_classes=num_classes, arch=arch)
    model = model.to(device)
    
    
    def calculate_accuracy(model, data_loader, device):
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient calculation
            for images, labels in data_loader:
                images = images.view(-1, T*C).to(device)  # Flatten the image
                labels = labels.to(device)
                outputs = model(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    optim= torch.optim.Adam(model.parameters(), lr=0.001)
    
    for iter in range(iterations):
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, T*C).to(device) #Flatten the image
            labels = labels.to(device)
            optim.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optim.step()
            # if (i+1) % 100 == 0:
            #     print(f'Iteration: {iter+1}, Batch={i+1}, Loss: {loss.item()}')
        if (iter)%5 ==0:
            train_accuracy = calculate_accuracy(model, train_loader, device)
            test_accuracy = calculate_accuracy(model, test_loader, device)
            print(f'Iteration: {iter+1}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')
            #Return to the client
            return jsonify({'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy})



if __name__ == '__main__':
    app.run(port=5000)