from ganData import create_dataloader
from model import create_model
from visualizer import Visualizer
import copy
import time
import os
import torch
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from tqdm import tqdm

device = 'cuda'
z_size = 50
class_num = 17
batch_size = 64
emotion_para_size = 53

def create_solver(opt):
    instance = Solver()
    instance.initialize(opt)
    return instance

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, labels):
    
    # Init gradient
    g_optimizer.zero_grad()
    
    # Building z
    z = Variable(torch.randn(batch_size, z_size)).to(device)
    
    # Building fake labels
    random_indices = torch.randint(0, len(labels), size=(batch_size,))
    fake_labels = labels[random_indices].unsqueeze(1).to(device)
    # fake_labels = torch.rand(batch_size, 1, 17).to(device)
    
    # Generating fake images
    fake_images = generator(z, fake_labels)
    
    # Disciminating fake images
    validity = discriminator(fake_images, fake_labels)
    
    # Calculating discrimination loss (fake images)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    
    # Backword propagation
    g_loss.backward()
    
    #  Optimizing generator
    g_optimizer.step()
    
    return g_loss.data

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_emotion_paras, labels):
    
    # Init gradient 
    d_optimizer.zero_grad()

    # Disciminating real images
    real_validity = discriminator(real_emotion_paras, labels)
    
    # Calculating discrimination loss (real images)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
    
    # Building z
    z = Variable(torch.randn(batch_size, z_size)).to(device)
    
    # Building fake labels
    # fake_labels = Variable(torch.LongTensor(np.random.rand(1, class_num, batch_size))).to(device)
    # print('-----------len(labels)-------------')
    # print(len(labels))
    # print(labels)
    
    random_indices = torch.randint(0, len(labels), size=(batch_size,))
    fake_labels = labels[random_indices].unsqueeze(1).to(device)
    
    # print('-----------fake_labels-------------')
    # print(len(fake_labels))
    # print(fake_labels)
    # fake_labels = torch.rand(batch_size, 1, 17).to(device)
    
    # Generating fake images
    fake_images = generator(z, fake_labels)    
    
    # Disciminating fake images
    fake_validity = discriminator(fake_images, fake_labels)
    
    # Calculating discrimination loss (fake images)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
    
    # Sum two losses
    d_loss = real_loss + fake_loss
    
    # Backword propagation
    d_loss.backward()
    
    # Optimizing discriminator
    d_optimizer.step()
    
    return d_loss.data

class Generator(nn.Module):
    def __init__(self, generator_layer_size, z_size, emotion_para_size, class_num):
        super().__init__()
        
        # self.label_emb = nn.Embedding(class_num, class_num)
        
        self.model = nn.Sequential(
            nn.Linear(z_size + class_num, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], generator_layer_size[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[3], generator_layer_size[4]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[4], emotion_para_size)
            # nn.Tanh()
        )

    
    def forward(self, z, labels):
        
        # Reshape z
        z = z.view(-1, z_size)
        
        labels = np.squeeze(labels)
        # Concat image & label
        x = torch.cat([z, labels], 1).float()
        
        # Generator out
        out = self.model(x)
        
        return out.view(-1, emotion_para_size)

class Discriminator(nn.Module):
    def __init__(self, discriminator_layer_size, emotion_para_size, class_num):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Tanh(),
            nn.Linear(emotion_para_size + class_num, discriminator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[0], discriminator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[1], discriminator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(discriminator_layer_size[2], 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        # Reshape fake image
        x = x.view(-1, emotion_para_size)
        c = np.squeeze(labels)
        # Concat image & label
        x = torch.cat([x, c], 1).float()
        out = self.model(x)
        return out.squeeze() 
    
    
    
class Solver(object):
    """docstring for Solver"""
    def __init__(self):
        super(Solver, self).__init__()

    def initialize(self, opt):
        self.opt = opt

    def run_solver(self):
        if self.opt.mode == "train":
            self.train_networks()
        else:
            self.test_networks(self.opt)

    def train_networks(self):
        # init train setting
        self.init_train_setting()
        # Loss function
        # criterion = nn.MSELoss()
        criterion = nn.BCELoss()
        epochs = self.epochs
        device = self.device
        
        for epoch in range(epochs):
            print('Starting epoch {}...'.format(epoch+1))
        
            for i, (images, labels) in tqdm(enumerate(self.train_dataset)):
                # Train data
                
                real_emotion_paras = Variable(images).to(device)
                # print(real_emotion_paras)
                labels = Variable(labels).to(device)
                # print(labels)
                # Set generator train
                self.generator.train()
        
                # Train discriminator
                d_loss = discriminator_train_step(len(real_emotion_paras), self.discriminator,
                                                  self.generator, self.d_optimizer, criterion,
                                                  real_emotion_paras, labels)
        
                # Train generator
                g_loss = generator_train_step(batch_size, self.discriminator, self.generator, self.g_optimizer, criterion, labels)
                
                # if i % 10 == 0:
                #     print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
            
            # Set generator eval
            self.generator.eval()
            print('g_loss: {}, d_loss: {}'.format(g_loss, d_loss))
            
            # Building z 
            # z = Variable(torch.randn(1, z_size)).to(device)
            # Building z 
            random_indices = torch.randint(0, len(labels), size=(2,))
            
            # z = labels[random_indices].unsqueeze(1).to(device)
            
            z = Variable(torch.randn(2, z_size)).to(device)
            
            labels = labels[random_indices].unsqueeze(1).to(device)
            real = images[random_indices].unsqueeze(1).to(device)
            
            # Generating images
            sample_images = self.generator(z, labels).unsqueeze(1).data.cpu()
            
            # print('-------------------labels-------------------------')
            # print(labels)
            print('--------------------------------------labels---------------------------------------------')
            print(labels)
            print('--------------------------------sample FRAME param---------------------------------------')
            print(sample_images)
            print('---------------------------------real FRAME param----------------------------------------')
            print(real)
            print('-----------------------------------------------------------------------------------------')
        
        torch.save(self.generator,'30_generator.pt')

    def init_train_setting(self):
        self.train_dataset = create_dataloader(self.opt)
        train_dataset = self.train_dataset
        
        batch_size = 64  # Batch size
        class_num = 17
        # Model
        z_size = 50

        self.epochs = 30  # Train epochs
        learning_rate = 2 * 1e-4

        generator_layer_size = [32, 64, 128, 256, 512]
        discriminator_layer_size = [128, 64, 32]
        
        self.device = 'cuda'
        
        device = self.device
        # Define generator
        self.generator = Generator(generator_layer_size, z_size, emotion_para_size, class_num).to(device)
        # Define discriminator
        self.discriminator = Discriminator(discriminator_layer_size, emotion_para_size, class_num).to(device)
        # Loss function
        self.criterion = nn.BCELoss()
        
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # t = testModel().to(device)
        print(self.generator)
        print(self.discriminator)
        
        # Optimizer
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)
        
    def test_networks(self, opt):
        self.init_test_setting(opt)
        self.test_ops()

    def init_test_setting(self, opt):
        self.test_dataset = create_dataloader(opt)
        
    