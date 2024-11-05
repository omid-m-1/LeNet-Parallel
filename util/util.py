import torch
from torch import distributed as dist, optim
from torch.nn import init as init, functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import torchvision.transforms as transforms

from util.model import LeNet

import time
import os


class Partition(object):
    """Custom Partition class to access a subset of data"""
    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """Custom DataPartitioner class to split data for each GPU."""
    def __init__(self, data, sizes, seed=42):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indices = list(range(data_len))
        rng.shuffle(indices)

        # Divide indices for each partition
        for size in sizes:
            part_len = int(size * data_len)
            self.partitions.append(indices[:part_len])
            indices = indices[part_len:]

    def use(self, partition):
        """Select the partition corresponding to a specific GPU rank."""
        return Partition(self.data, self.partitions[partition])

# Calculate mean and std
def calculate_mean_std(loader):
    total_sum = 0.0
    squared_sum = 0.0
    samples = 0

    for data in loader:
        images, _ = data
        # Reshape to (batch_size, 3, height * width)
        images = images.view(images.size(0), images.size(1), -1)
        # Calculate sum and sum of squared
        total_sum += images.mean(2).sum(0)
        squared_sum += (images**2).mean(2).sum(0)
        # Count samples
        samples += images.size(0)

    # Calculate mean and std
    mean = total_sum / samples
    std = torch.sqrt(squared_sum / samples - mean**2)

    return mean, std

def load_data(train=True, shuffle=False, download=True, num_workers=1, rank=0, total_ranks=1, batch_size=64, seed=42, transform=None):
    '''MNIST data loader'''
    if transform == None:
        transform = transforms.Compose([transforms.ToTensor()])
    data = torchvision.datasets.MNIST(root='./data', train=train, download=download, transform=transform)
    if (total_ranks > 1):
        # partitioning without overlapping
        #data = random_split(data, [len(data)//total_ranks] * total_ranks)
        partition_sizes = [1.0 / total_ranks] * total_ranks #[1.0 / total_ranks for _ in range(total_ranks)]
        partitioner = DataPartitioner(data, partition_sizes, seed=seed)
        data = partitioner.use(rank)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader

def partition_dataset(batch_size, mean, std, rank, total_ranks, num_cores, seed):
    '''Partition data for GPUs'''
    # Data augmentation and normalization with horizontal filip and random crops
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Train data loader
    trainloader = load_data(train=True, shuffle=True, download=False, num_workers=num_cores, 
        rank=rank, total_ranks=total_ranks, batch_size=batch_size, transform=transform_train)

    # Test data loader
    testloader = load_data(train=False, shuffle=False, download=False, num_workers=num_cores, 
        rank = rank, total_ranks=total_ranks, batch_size=batch_size, transform=transform_test)
    return trainloader, testloader

def average_gradients(model):
    """Average gradients across all GPUs."""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.data /= dist.get_world_size()

def evaluate_model(model, test_loader, device, rank, total_ranks, epochs, epoch=-1, mixed=False):
    """Evaluate the model using all GPUs."""
    model.eval()
    test_loss = 0.0
    correct, samples = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if mixed:
                with autocast():
                    # Forward pass
                    outputs = model(inputs)
                    loss = F.nll_loss(outputs, labels)
            else:
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels)

            # Compute loss and accuracy
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            samples += labels.size(0)
            correct += (predicted==labels).sum().item()

    # reduce results
    # Gather results from all GPUs
    if (total_ranks > 1):
        correct = torch.tensor(correct).to(device)
        samples = torch.tensor(samples).to(device)
        test_loss = torch.tensor(test_loss).to(device)
        dist.reduce(correct, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(samples, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(test_loss, dst=0, op=dist.ReduceOp.SUM)

    if (rank == 0):
        test_accuracy = 100.0 * correct / samples
        test_loss = test_loss / samples
        if (epoch > -1): print(f'Epoch: [{epoch+1}/{epochs}] | Test Loss: {test_loss:.4f}, | Test Accuracy: {test_accuracy:.2f}%')
        else: print(f'Test Loss: {test_loss:.4f}, | Test Accuracy: {test_accuracy:.2f}%')

def train_model(model, train_loader, test_loader, optimizer, device, rank, total_ranks, epochs, mixed=False, test=False):
    '''Train the model on multi GPUs.'''
    # Main training loop
    for epoch in range(epochs):
        # Set train mode
        model.train()
        train_loss = 0.0
        correct, samples = 0, 0

        # Train each batch
        for inputs, labels in train_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            if mixed:
                with autocast():
                    # Forward pass
                    outputs = model(inputs)
                    loss = F.nll_loss(outputs, labels)
                # Backward pass
                scaler.scale(loss).backward()
                average_gradients(model)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                outputs = model(inputs)
                #loss = F.cross_entropy(outputs, labels)
                loss = F.nll_loss(outputs, labels)
                # Backward pass
                loss.backward()
                average_gradients(model)
                optimizer.step()

            # Compute loss and accuracy
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            samples += labels.size(0)
            correct += (predicted==labels).sum().item()

        train_accuracy = 100.0 * correct / samples
        train_loss = train_loss / samples

        if (total_ranks > 1): print(f"[GPU: {rank+1}] | Epoch: [{epoch+1}/{epochs}] | \
Loss: {train_loss:.4f}, | Accuracy: {train_accuracy:.2f}%")
        else: print(f"Epoch: [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, | Train Accuracy: {train_accuracy:.2f}%")

        if (test): evaluate_model(model, test_loader, rank, total_ranks, epochs, epoch=epoch, mixed=mixed)

def setup_and_train(kernel, rank, total_ranks, batch_size, epochs, learning_rate, mixed, use_ddp = False, test=False, seed=42, backend='nccl'):
    """Initialize the distributed training environment."""
    # Set tensor cores for mixed percission
    if (mixed == 'mixed_tc'): torch.set_float32_matmul_precision('medium')
    # Set master address and port
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '27500'
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=total_ranks)

    # Set model and dataloader for GPUs
    device = torch.device(f'cuda:{rank}')
    # Initialize model, move it to appropriate device
    model = LeNet(kernel = kernel).to(device)
    if (use_ddp): model = DDP(model, device_ids=[rank])  # Wrap model in DDP for distributed training
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    num_cores = 4 #os.cpu_count()

    # Calculate mean and std of the dataset
    if (rank == 0):
        trainloader = load_data(batch_size=100, seed=seed)
        mean, std = calculate_mean_std(trainloader)
    else:
        mean, std = torch.tensor(0.0), torch.tensor(0.0)
    if (backend=='nccl'): mean, std = mean.to(device), std.to(device)
    torch.cuda.synchronize(device)
    dist.broadcast(mean, src=0)
    dist.broadcast(std, src=0)
    if (backend=='nccl'):
        mean = mean.cpu()
        std = std.cpu()

    # Load training and test data for this gpu rank
    train_loader, test_loader = partition_dataset(batch_size, mean, std, rank, total_ranks, num_cores, seed=seed)

    # Train model
    train_model(model, train_loader, test_loader, optimizer, device, rank, total_ranks, epochs, mixed=(mixed=='mixed_amp'), test=test)

    # Test Model
    evaluate_model(model, test_loader, device, rank, total_ranks, epochs, epoch=-1, mixed=(mixed=='mixed_amp'))

    # Clean up process group
    dist.destroy_process_group()
