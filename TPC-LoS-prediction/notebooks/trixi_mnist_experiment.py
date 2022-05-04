import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms

from trixi.util import Config
from trixi.experiment import PytorchExperiment
import os


# build a simple cnn model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MNIST_experiment(PytorchExperiment):
    def setup(self):

        self.elog.print("Config:")
        self.elog.print(self.config)

        ### Get Dataset
        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset_train = datasets.MNIST(root="experiment_dir/data/", download=True,
                                            transform=transf, train=True)
        self.dataset_test = datasets.MNIST(root="experiment_dir/data/", download=True,
                                           transform=transf, train=False)

        data_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if self.config.use_cuda else {}

        self.train_data_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.config.batch_size,
                                                             shuffle=True, **data_loader_kwargs)
        self.test_data_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=self.config.batch_size,
                                                            shuffle=True, **data_loader_kwargs)


        self.device = torch.device("cuda" if self.config.use_cuda else "cpu")

        self.model = Net()
        self.model.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate,
                                   momentum=self.config.momentum)

        self.save_checkpoint(name="checkpoint_start")
        self.vlog.plot_model_structure(self.model,
                                       [self.config.batch_size, 1, 28, 28],
                                       name='Model Structure')

        self.batch_counter = 0
        self.elog.print('Experiment set up.')


    def train(self, epoch):

        self.model.train()

        for batch_idx, (data, target) in enumerate(self.train_data_loader):

            self.batch_counter += 1

            if self.config.use_cuda:
                data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()

            output = self.model(data)
            self.loss = F.nll_loss(output, target)
            self.loss.backward()

            self.optimizer.step()

            if batch_idx % self.config.log_interval == 0:
                # plot train loss (mathematically mot 100% correct, just so that lisa can sleep at night (if no one is breathing next to her ;-P) )
                self.add_result(value=self.loss.item(), name='Train_Loss',
                                counter=epoch + batch_idx / len(self.train_data_loader), label='Loss')
                # log train batch loss and progress
                self.clog.show_text(
                    'Train Epoch: {} [{}/{} samples ({:.0f}%)]\t Batch Loss: {:.6f}'
                        .format(epoch, batch_idx * len(data),
                                len(self.train_data_loader.dataset),
                                100. * batch_idx / len(self.train_data_loader),
                                self.loss.item()), name="log")

                self.clog.show_image_grid(data, name="mnist_training", n_iter=epoch + batch_idx / len(self.train_data_loader), iter_format="{:0.02f}")

                self.save_checkpoint(name="checkpoint", n_iter=batch_idx)

    def validate(self, epoch):
        self.model.eval()

        validation_loss = 0
        correct = 0

        for data, target in self.test_data_loader:
            if self.config.use_cuda:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            validation_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        validation_loss /= len(self.test_data_loader.dataset)
        # plot the test loss
        self.add_result(value=validation_loss, name='Validation_Loss',
                        counter=epoch + 1, label='Loss')
        # plot the test accuracy
        acc = 100. * correct / len(self.test_data_loader.dataset)
        self.add_result(value=acc, name='ValidationAccurracy',
                        counter=epoch + 1, label='Accurracy' )

        # log validation loss and accuracy
        self.elog.print(
            '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                .format(validation_loss, correct, len(self.test_data_loader.dataset),
                        100. * correct / len(self.test_data_loader.dataset)))


def main():
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

    c = Config()

    c.batch_size = 64
    c.batch_size_test = 1000
    c.n_epochs = 10
    c.learning_rate = 0.01
    c.momentum = 0.9
    if torch.cuda.is_available():
        c.use_cuda = True
    else:
        c.use_cuda = False
    c.rnd_seed = 1
    c.log_interval = 200
    exp = MNIST_experiment(config=c, name='experiment', n_epochs=c.n_epochs,
                           seed=42, base_dir='./experiment_dir',
                           loggers={"visdom": "visdom"})
    exp.run()


if __name__ == '__main__':
    main()
