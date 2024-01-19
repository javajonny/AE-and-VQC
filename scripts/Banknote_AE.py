#!/usr/bin/env python
# coding: utf-8

# ## Setup

# In[ ]:


import time
import random 
import pennylane as qml
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import os
import csv
import sys


print("+"*50)
print("Banknote Authentication")
print("+"*50)
print("\n")


print(f"PennyLane version = {qml.version()}")
print(f"Pytorch version = {torch. __version__ }")

setgrad = lambda g, *ms: [setattr(p,'requires_grad', g) for m in ms for p in m.parameters() ]


SYS_SEED = int(sys.argv[1])    # Seed for random initial weights
SYS_BATCH_SIZE =  int(sys.argv[2])
SYS_LEARNING_RATE = float(sys.argv[3])
SYS_LAYERS = int(sys.argv[4])

EPOCHS_SETTING = 100

# REPRODUCIBILITY 
SEED = SYS_SEED   # Seed for random initial weights
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# AE
LEARNING_RATE_AE = 0.1
BATCH_SIZE_AE = 128
EPOCHS_AE = 500
ENCODER_ACTIVATION_FN = nn.Sigmoid()
DECODER_ACTIVATION_FN = nn.Sigmoid()

# VQC with Angle Embedding
LEARNING_RATE_VQC = 0.01
BATCH_SIZE_VQC =  SYS_BATCH_SIZE
EPOCHS_VQC = EPOCHS_SETTING
LAYERS_ANGLE_EMBEDDING = SYS_LAYERS

# VQC with Amplitude Embedding
LEARNING_RATE_VQC_AMPLITUDE_EMBEDDING = 0.01
BATCH_SIZE_VQC_AMPLITUDE_EMBEDDING =  SYS_BATCH_SIZE
EPOCHS_VQC_AMPLITUDE_EMBEDDING = EPOCHS_SETTING
LAYERS_AMPLITUDE_EMBEDDING = SYS_LAYERS

# NN on original Input
LEARNING_RATE_ONLY_NN = 0.1
BATCH_SIZE_ONLY_NN = SYS_BATCH_SIZE
EPOCHS_ONLY_NN = EPOCHS_SETTING

# NN on compressed Input (with AE before) 
LEARNING_RATE_AE_NN = 0.1
BATCH_SIZE_AE_NN = SYS_BATCH_SIZE
EPOCHS_AE_NN = EPOCHS_SETTING

# SEQUENT
PREPROCESSING_SEQUENT_ACTICATION_FN = nn.Sigmoid()
LEARNING_RATE_SEQUENT = 0.1
BATCH_SIZE_SEQUENT = SYS_BATCH_SIZE
EPOCHS_SEQUENT = EPOCHS_SETTING
LAYERS_SEQUENT = SYS_LAYERS

# DRESSED
PREPROCESSING_DRESSED_ACTICATION_FN = nn.Sigmoid()
LEARNING_RATE_DRESSED = 0.1
BATCH_SIZE_DRESSED = SYS_BATCH_SIZE
EPOCHS_DRESSED = EPOCHS_SETTING
LAYERS_DRESSED = SYS_LAYERS


print('-' * 50)
print('-' * 50)
print('-' * 50)

# Print the information
print("Reproducibility:")
print(f"Seed: {SEED}")
print("\nAE:")
print(f"- Learning Rate: {LEARNING_RATE_AE}")
print(f"- Batch Size: {BATCH_SIZE_AE}")
print(f"- Epochs: {EPOCHS_AE}")
print(f"- Encoder Activation Function: {ENCODER_ACTIVATION_FN}")
print(f"- Decoder Activation Function: {DECODER_ACTIVATION_FN}")
print("\nVQC with Angle Embedding:")
print(f"- Learning Rate: {LEARNING_RATE_VQC}")
print(f"- Batch Size: {BATCH_SIZE_VQC}")
print(f"- Epochs: {EPOCHS_VQC}")
print(f"- Layers: {LAYERS_ANGLE_EMBEDDING}")
print("\nVQC with Amplitude Embedding:")
print(f"- Learning Rate: {LEARNING_RATE_VQC_AMPLITUDE_EMBEDDING}")
print(f"- Batch Size: {BATCH_SIZE_VQC_AMPLITUDE_EMBEDDING}")
print(f"- Epochs: {EPOCHS_VQC_AMPLITUDE_EMBEDDING}")
print(f"- Layers: {LAYERS_AMPLITUDE_EMBEDDING}")
print("\nNN on Original Input:")
print(f"- Learning Rate: {LEARNING_RATE_ONLY_NN}")
print(f"- Batch Size: {BATCH_SIZE_ONLY_NN}")
print(f"- Epochs: {EPOCHS_ONLY_NN}")
print("\nNN on Compressed Input (with AE before):")
print(f"- Learning Rate: {LEARNING_RATE_AE_NN}")
print(f"- Batch Size: {BATCH_SIZE_AE_NN}")
print(f"- Epochs: {EPOCHS_AE_NN}")
print("\nSequent:")
print(f"- Preprocessing Activation Function: {PREPROCESSING_SEQUENT_ACTICATION_FN}")
print(f"- Learning Rate: {LEARNING_RATE_SEQUENT}")
print(f"- Batch Size: {BATCH_SIZE_SEQUENT}")
print(f"- Epochs: {EPOCHS_SEQUENT}")
print(f"- Layers: {LAYERS_SEQUENT}")
print("\nDressed Quantum Circuit:")
print(f"- Preprocessing Activation Function: {PREPROCESSING_DRESSED_ACTICATION_FN}")
print(f"- Learning Rate: {LEARNING_RATE_DRESSED}")
print(f"- Batch Size: {BATCH_SIZE_DRESSED}")
print(f"- Epochs: {EPOCHS_DRESSED}")
print(f"- Layers: {LAYERS_DRESSED}")


print('-' * 50)
print('-' * 50)
print('-' * 50)


# ## Getting data ready

# In[ ]:


class BankNoteDataset(Dataset):
    def __init__(self, file_path, train=False, validation=False, test=False, seed=SEED):
        # load data into dataframe
        self.dataframe = pd.read_csv(file_path)

        # Drop rows with missing values
        self.dataframe = self.dataframe.dropna()

        # Convert the dataset to numpy arrays
        data = self.dataframe.to_numpy()

        # Split the dataset into X and y
        X = data[:, :-1]
        y = data[:, -1]

        # Store the input dimension of X
        self.input_dimension = X.shape[1]

        # Store the feature names
        self.feature_names = self.dataframe.columns[:4].tolist()

        # Store the label name
        self.label_names = self.dataframe.columns[-1]

        # Convert y to integer labels
        y = y.astype(int)

        # Apply one-hot encoding to y
        self.num_classes = len(np.unique(y))
        y = np.eye(self.num_classes)[y]

        # Apply Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)
        
        # Split: 80%, 10%, 10%
        # here it makes no difference if we split the train-set into training & validation or the test-set because it comes from the same dataset 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed, shuffle=True)

        if train:
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train, dtype=torch.float32)
        elif validation:
            self.X = torch.tensor(X_val, dtype=torch.float32)
            self.y = torch.tensor(y_val, dtype=torch.float32)     
        elif test:
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.float32)     

       
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y 

    def visualize_output_distribution(self):
        # Count the occurrences of each class
        class_counts = self.y.sum(dim=0)

        # Get the class labels
        class_labels = [str(i) for i in range(self.num_classes)]

        # Plot the output distribution
        plt.bar(class_labels, class_counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Output Distribution')
        plt.show()


# In[ ]:


file_path = os.path.join(os.getcwd(), '..', 'data', 'BankNote_Authentication.csv')

banknote_train = BankNoteDataset(file_path, train=True)
banknote_validation = BankNoteDataset(file_path, validation=True)
banknote_test = BankNoteDataset(file_path, test=True)


num_classes = banknote_train.num_classes
input_dimension = banknote_train.input_dimension


# angle embedding: Encodes N features into the rotation angles of n qubits, where N≤n
VQC_width = num_classes
# amplitude embedding: Encodes 2^n features into the amplitude vector of n qubits.
wires_amplitude = max(math.ceil(math.log(input_dimension, 2)), num_classes) #at least as many qubits as output classes


# In[ ]:


len(banknote_train), len(banknote_validation), len(banknote_test)


# In[ ]:


X_train_max = torch.max(banknote_train.X)
X_train_min = torch.min(banknote_train.X)
X_test_max = torch.max(banknote_test.X)
X_test_min = torch.min(banknote_test.X)


print(f"Maximum of X_train: {X_train_max}")
print(f"Minimum of X_train: {X_train_min}")
print(f"Maximum of X_test: {X_test_max}")
print(f"Minimum of X_test: {X_test_min}")


# In[ ]:


banknote_train.visualize_output_distribution()
banknote_validation.visualize_output_distribution()
banknote_test.visualize_output_distribution()


# ## Models

# In[ ]:


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_last_layer_activation=ENCODER_ACTIVATION_FN, decoder_last_layer_activation=DECODER_ACTIVATION_FN):
        super().__init__()

        # calculate the number of neurons for each layer
        neuron_list = []
        running_size = input_size
        while(running_size > hidden_size):
            neuron_list.append(running_size)
            running_size = running_size // 2
        neuron_list.append(hidden_size)

        # Encoder layers
        encoder_layers = []
        length = len(neuron_list)
        for i in range(length-1):
            encoder_layers.append(nn.Linear(neuron_list[i], neuron_list[i+1]))
            if(i != length-2):
                encoder_layers.append(nn.ReLU())
            else:
                encoder_layers.append(encoder_last_layer_activation)

        self.encoder = nn.Sequential(*encoder_layers) # asterisk (*) operator to unpack the list into separate arguments


        # Decoder layers
        decoder_layers = []
        for i in range(length - 1, 0, -1):
            decoder_layers.append(nn.Linear(neuron_list[i], neuron_list[i-1]))
            if(i != 1):
                decoder_layers.append(nn.ReLU())
            else:
                decoder_layers.append(decoder_last_layer_activation)

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# Sigmoid activation function to get values between 0 and 1


# In[ ]:


"""Layer of single-qubit Hadamard gates. """
def Hadamard(nqubits):
    return [qml.Hadamard(wires=idx) for idx in range(nqubits)]
        
"""Layer of parametrized qubit rotations around the y axis."""
def Rotation(w):
    return [qml.RY(element, wires=idx) for idx, element in enumerate(w)]

"""Layer of shifted CNots."""
def CNot(start, nqubits):
    return [qml.CNOT(wires=[i, i + 1]) for i in range(start, nqubits - 1, 2)] 

"""Layer of CNOTs followed by another shifted layer of CNOTs and a Rotation Layer"""   
def Entangle(weights): 
    return [[*CNot(0, len(w)), *CNot(1, len(w)), *Rotation(w)] for w in weights]

"""Expectation values in the Z basis."""
def Measure(wires):
    return [qml.expval(qml.PauliZ(position)) for position in wires]  



dev_angle_embedding = qml.device('lightning.qubit', wires=VQC_width)
dev_amplitude_embedding = qml.device('lightning.qubit', wires=wires_amplitude)


@qml.qnode(dev_angle_embedding, interface="torch", diff_method='adjoint')
def variational_circuit_angle_embedding(input, weights, out):
    weights =  2.0 * torch.arctan(2 * weights) # weight remapping
    width = weights.shape[1]    
    assert input.shape[0] == width, f"Expected input of len {width}"
    input = input * np.pi - np.pi / 2.0   # Rescale [0, 1] to [-pi/2, pi/2]
    Hadamard(width)               # Start from state |+> , unbiased w.r.t. |0> and |1>
    Rotation(input)               # Embed features in the quantum node 
    Entangle(weights)             # Sequence of trainable variational layers
    return Measure(range(out))    # Expectation values in the Z basis



@qml.qnode(dev_amplitude_embedding, interface="torch", diff_method='adjoint')
def variational_circuit_amplitude_embedding(input, weights, out):
    torch_pi = torch.tensor(math.pi)
    weights =  torch_pi * torch.tanh(weights) # weight remapping
    width = weights.shape[1]    
    input = input.tolist()
    qml.AmplitudeEmbedding(features=input, wires=range(width), normalize=True, pad_with=0.)  # Embed features in the quantum node
    Entangle(weights)             # Sequence of trainable variational layers
    return Measure(range(out))    # Expectation values in the Z basis



class Circuit(nn.Module):
    def __init__(self, width, depth, out, amplitude_embedding):
        super().__init__()
        self.out = out
        self.params = torch.nn.Parameter(torch.randn(depth, width))
        self.amplitude_embedding = amplitude_embedding

    def forward(self, input):
        if len(input.shape) > 1: return torch.cat([self(i).float().unsqueeze(0) for i in input])
        if self.amplitude_embedding:
            return variational_circuit_amplitude_embedding(input, self.params, self.out).float()
        else:
            return variational_circuit_angle_embedding(input, self.params, self.out).float()



class ClassicalNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# In[ ]:


class DressedQuantumCircuit(torch.nn.Module):
  def __init__(self, width, depth, i=input_dimension, o=num_classes):
    """ :param i,o: Input, Output dimension, :params width, depth: Internal net width (i.e. n_qubits) and depth (number of variational layers)"""
    super().__init__()
    self.pre_processing  = torch.nn.Sequential(torch.nn.Linear(i, width), PREPROCESSING_DRESSED_ACTICATION_FN) 
    self.circuit = Circuit(width, depth, width, amplitude_embedding=False)
    self.post_processing = torch.nn.Linear(width, o)
  
  def train(self, mode): 
    if mode == 'classical': setgrad(True, self.pre_processing, self.post_processing); setgrad(False, self.circuit)
    if mode == 'quantum': setgrad(True, self.circuit); setgrad(False, self.pre_processing, self.post_processing)
    if mode == 'hybrid': setgrad(True, self.pre_processing, self.circuit, self.post_processing)

  def forward(self, input): return self.post_processing(self.circuit(self.pre_processing(input.float())))


# In[ ]:


class SEQUENT(torch.nn.Module):
  """ Sequential Quantum Enhanced Training (SEQUENT) """
  def __init__(self, width, depth, i=input_dimension, o=num_classes):
    """ :params i,o: Input, Output dimension, :params width, depth: Internal net width (i.e. n_qubits) and depth (number of variational layers)"""
    super().__init__()
    self.compression  = torch.nn.Sequential(torch.nn.Linear(i, width),PREPROCESSING_SEQUENT_ACTICATION_FN) 
    self.surrogate = torch.nn.Sequential(torch.nn.Linear(width, o))
    self.circuit = Circuit(width, depth, o, amplitude_embedding=False)

  def train(self, mode): 
    if mode == 'classical': self.classification = self.surrogate; setgrad(False, self.circuit); setgrad(True, self.compression, self.surrogate)
    if mode == 'quantum': self.classification = self.circuit; setgrad(True, self.circuit); setgrad(False, self.compression, self.surrogate)
    if mode == 'hybrid': self.classification = self.circuit; setgrad(True, self.compression, self.circuit); setgrad(False, self.surrogate)

  def forward(self, input): return self.classification(self.compression(input.float()))


# ## Testing Functions

# In[ ]:


def test_autoencoder(model, dataset_test, batch_size):
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    test_loss = 0.0

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        for original_images_val, _ in dataloader_test:
            recon_images_val = model(original_images_val)
            loss = criterion(recon_images_val, original_images_val)
            test_loss += loss.item() * original_images_val.shape[0] 

    # Compute average test loss for the epoch
    test_loss /= len(dataloader_test.dataset)

    return test_loss


# In[ ]:


def test_model(model, dataset_test, batch_size):
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    model.eval()
    results_list = []
    test_loss = 0.0
    total_accuracy = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader_test:
            test_predictions = model(batch_inputs)
            y_probs = torch.softmax(test_predictions, dim=1)
            y_preds = torch.argmax(y_probs, dim=1)
            y_trues = torch.argmax(batch_labels, dim=1)
            
            for i in range(len(batch_inputs)):
                sample_result = (y_trues[i].item(), y_preds[i].item(), y_probs[i].tolist())
                results_list.append(sample_result)

            # Compute the test loss
            loss = criterion(test_predictions, y_trues)
            test_loss += loss.item() * batch_inputs.shape[0]

            # Compute the test accuracy
            total_accuracy += (y_preds == y_trues).sum().item()

    # Calculate average test loss and test accuracy
    avg_test_loss = test_loss / len(dataset_test)
    avg_test_accuracy = total_accuracy / len(dataset_test)


    return results_list, avg_test_loss, avg_test_accuracy


# ## Training Functions

# In[ ]:


def train_and_validate_autoencoder(model, dataset_train, dataset_validation, batch_size, num_epochs, learning_rate):
    # Define the dataloader for efficient batch training
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_values = []

    # Train the model for the specified number of epochs
    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0

        start_time = time.time()

        # Train the model
        model.train()
        for original_images, _ in dataloader_train:
            optimizer.zero_grad()
            recon_image = model(original_images)
            loss = criterion(recon_image, original_images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * original_images.size(0) # loss.item() * batch_size

        # Compute average training loss for the epoch
        train_loss /= len(dataloader_train.dataset)

        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            for original_images_val, _ in dataloader_validation:
                recon_images_val = model(original_images_val)
                loss = criterion(recon_images_val, original_images_val)
                val_loss += loss.item() * original_images_val.size(0) 

        # Compute average validation loss for the epoch
        val_loss /= len(dataloader_validation.dataset)

        end_time = time.time()
        epoch_time = end_time - start_time

        loss_values.append((epoch+1, train_loss, val_loss, epoch_time))
        
        # print out
        print(f"Epoch: [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Epoch Time: {epoch_time:.2f} seconds")

    return loss_values


# In[ ]:


def show_parameters(model):
    for p in model.parameters():
        print(p)
        print(f"p.shape: {p.shape}")

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 
    #numel() in order to calculate the total number of elements in a PyTorch tensor or parameter.


def train_and_validate_model(model, dataset_train, dataset_validation, batch_size, num_epochs, learning_rate):    
    # Define the dataloader for efficient batch training
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)
    
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    result_list = []

    for epoch in range(num_epochs):
        # Initialize the total loss and accuracy for this epoch
        total_loss = 0.0
        total_accuracy = 0.0

        start_time = time.time()

        # Train the model
        model.train()

        # Loop over the batches
        for batch_inputs, batch_labels in dataloader_train:
            # Reset the gradients
            optimizer.zero_grad()
            # Compute the predictions
            train_predictions = model(batch_inputs)
            # Compute the loss
            loss = criterion(train_predictions, batch_labels.argmax(dim=1))
            # Accumulate the loss and accuracy
            total_loss += loss.item() * batch_inputs.shape[0]
            total_accuracy += (train_predictions.argmax(axis=1) == batch_labels.argmax(axis=1)).float().sum().item()
            # Compute the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()
        
        # Compute the average loss and accuracy for this epoch
        avg_train_loss = total_loss / len(dataset_train)
        avg_train_accuracy = total_accuracy / len(dataset_train)

        # Evaluate the model on the validation set
        model.eval()
        total_validation_loss = 0.0
        total_validation_accuracy = 0.0

        with torch.no_grad():
            for batch_inputs, batch_labels in dataloader_validation:
                val_predictions = model(batch_inputs)
                val_loss = criterion(val_predictions, batch_labels.argmax(dim=1))
                total_validation_loss += val_loss.item() * batch_inputs.shape[0]
                total_validation_accuracy += (val_predictions.argmax(axis=1) == batch_labels.argmax(axis=1)).float().sum().item()
        
        # Compute the average validation loss and accuracy for this epoch
        avg_val_loss = total_validation_loss / len(dataset_validation)
        avg_val_accuracy = total_validation_accuracy / len(dataset_validation)

        end_time = time.time()
        epoch_time = end_time - start_time
        # Print the progress
        print(f"Epoch: {epoch + 1}/{num_epochs}, Train loss = {avg_train_loss:.4f}, Validation loss = {avg_val_loss:.4f}, Train accuracy = {avg_train_accuracy:.4f},  Validation accuracy = {avg_val_accuracy:.4f}, Epoch Time: {epoch_time:.2f} seconds")
        
        # Add values to list
        result_list.append((epoch+1, avg_train_loss, avg_val_loss, avg_train_accuracy, avg_val_accuracy, epoch_time))
    
    return result_list


# In[ ]:


def train_and_validate_SEQUENT_DRESSED(model, dataset_train, dataset_validation, dataset_test, stages, num_epochs, batch_size, learning_rate):
    print('-' * 50)
    print('-' * 50)
    print(f'Model: {model.__class__.__name__}')
    print(f"epochs: {num_epochs}")
    print(f"batch size: {batch_size}")
    print(f"learning rate: {learning_rate}")
    print('-' * 50)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=True)

    result_list = []
    test_result_list = []

    for stage in stages:
        print(f"Training mode: {stage}")
        result_list_stage = []

        for epoch in range(num_epochs):
            model.train(stage) # defines which part should be trained
            total_loss = 0.0
            total_accuracy = 0.0

            start_time = time.time()

            # Loop over the batches
            for batch_inputs, batch_labels in dataloader_train:
                # Reset the gradients
                optimizer.zero_grad()
                # Compute the predictions
                train_predictions = model(batch_inputs)
                # Compute the loss
                loss = criterion(train_predictions, batch_labels.argmax(dim=1))
                # Accumulate the loss and accuracy
                total_loss += loss.item() * batch_inputs.shape[0]
                total_accuracy += (train_predictions.argmax(axis=1) == batch_labels.argmax(axis=1)).float().sum().item()
                # Compute the gradients
                loss.backward()
                # Update the parameters
                optimizer.step()

            # Compute the average loss and accuracy for this epoch
            avg_train_loss = total_loss / len(dataset_train)
            avg_train_accuracy = total_accuracy / len(dataset_train)
 

            # Evaluate the model on the validation set
            model.eval()
            total_validation_loss = 0.0
            total_validation_accuracy = 0.0

            with torch.no_grad():
                for batch_inputs, batch_labels in dataloader_validation:
                    val_predictions = model(batch_inputs)
                    val_loss = criterion(val_predictions, batch_labels.argmax(dim=1))
                    total_validation_loss += val_loss.item() * batch_inputs.shape[0]
                    total_validation_accuracy += (val_predictions.argmax(axis=1) == batch_labels.argmax(axis=1)).float().sum().item()
            
            # Compute the average validation loss and accuracy for this epoch
            avg_val_loss = total_validation_loss / len(dataset_validation)
            avg_val_accuracy = total_validation_accuracy / len(dataset_validation)

            end_time = time.time()
            epoch_time = end_time - start_time

            # Print the progress
            print(f"Model: {model.__class__.__name__}, Stage: {stage} --- Epoch: {epoch + 1}/{num_epochs}, Train loss = {avg_train_loss:.4f}, Validation loss = {avg_val_loss:.4f}, Train accuracy = {avg_train_accuracy:.4f},  Validation accuracy = {avg_val_accuracy:.4f}, Epoch Time: {epoch_time:.2f} seconds")

            # Add values to list
            result_list_stage.append((epoch+1, avg_train_loss, avg_val_loss, avg_train_accuracy, avg_val_accuracy, epoch_time))

        # Add testing and validation values to list
        result_list.append((stage, result_list_stage))

        # Testing
        test_list_stage, avg_test_loss, avg_test_accuracy = test_model(model, dataset_test, batch_size)
        test_result_list.append((stage, test_list_stage, avg_test_loss, avg_test_accuracy))


    return result_list, test_result_list


# ## Train and Evaluate the Models

# ### Autoencoder

# In[ ]:


# Define the Autoencoder model 
autoencoder = Autoencoder(input_size=input_dimension, hidden_size=VQC_width, encoder_last_layer_activation=ENCODER_ACTIVATION_FN, decoder_last_layer_activation=DECODER_ACTIVATION_FN)
print("\nAutoencoder results")
# Train and validate the autoencoder model
AE_loss_values = train_and_validate_autoencoder(model = autoencoder, dataset_train=banknote_train, dataset_validation=banknote_validation, batch_size=BATCH_SIZE_AE, num_epochs=EPOCHS_AE, learning_rate=LEARNING_RATE_AE)

# Testing
AE_test_loss = test_autoencoder(autoencoder, dataset_test=banknote_test, batch_size=BATCH_SIZE_AE)
print(f"AE test loss: {AE_test_loss}")

# Test if AE classifies
AE_classification_testing_list, AE_classification_testing_loss, AE_classification_testing_accuracy = test_model(autoencoder, dataset_test=banknote_test, batch_size=BATCH_SIZE_AE)
print(AE_classification_testing_accuracy)


# ## Save the Results

# In[ ]:


# Define the hyperparameters
HYPERPARAMETERS = {
    "AE": {
        "SEED": SEED,
        "AE_LEARNING_RATE": LEARNING_RATE_AE,
        "AE_BATCH_SIZE": BATCH_SIZE_AE,
        "AE_EPOCHS": EPOCHS_AE,
        "AE_ENCODER_ACTIVATION_FN": str(ENCODER_ACTIVATION_FN),
        "AE_DECODER_ACTIVATION_FN": str(DECODER_ACTIVATION_FN),
    }
}


def save_list_to_csv(data_list, hyperparameters, list_name, autoencoder, testing, test_loss=None, test_accuracy=None):
    dataset_name = "BankNote_Authentication"
    header_list = []
    result_list = []
    header_list.append("Dataset")
    header_list.append("List Name")
    result_list.append(dataset_name)
    result_list.append(list_name)


    hyperparam_str = "_".join([f"{param}-{value}" for param, value in hyperparameters.items()])

    # Add the hyperparameters for the AE to all csv files
    AE_hyperparameters = HYPERPARAMETERS["AE"]
    for param, value in AE_hyperparameters.items():
        header_list.append(param)
        result_list.append(value)

    # Add specific hyperparameters 
    if(not autoencoder):
        for param, value in hyperparameters.items():
            header_list.append(param)
            result_list.append(value)

    # Create the file name by combining the list name, hyperparameters, and the ".csv" extension
    file_name = f"{dataset_name}--{list_name}_results_{hyperparam_str}.csv"

    # Save the list to the CSV file
    with open(file_name, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the column headers based on the type of result
        if autoencoder:
            header_list = header_list + ["Epoch", "Train_Loss", "Validation_Loss", "Time", "Test_Loss"]
            writer.writerow(header_list)
        elif testing:
            header_list = header_list + ["y_true", "y_pred", "y_probs", "Test_Loss", "Test_Accuracy"]
            writer.writerow(header_list)
        else:
            header_list = header_list + ["Epoch", "Train_Loss", "Validation_Loss", "Train_Accuracy", "Validation_Accuracy", "Time", "Test_Loss", "Test_Accuracy"]
            writer.writerow(header_list)
        
        result_list = [list(result_list) + list(data) for data in data_list]

        # if AE -> add the test loss
        if(autoencoder):
            result_list = [[*entry, test_loss] for entry in result_list]
        else: # add test loss and test accuracy
            result_list = [[*entry, test_loss, test_accuracy] for entry in result_list]

        # Write the data rows
        writer.writerows(result_list)
 

# Training & Validation lists to csv
save_list_to_csv(AE_loss_values, HYPERPARAMETERS["AE"], "AE_loss_values", autoencoder=True, testing=False, test_loss=AE_test_loss)

save_list_to_csv(AE_classification_testing_list, HYPERPARAMETERS["AE"], "AE_classification_testing_list", autoencoder=False, testing=True, test_loss=AE_classification_testing_loss, test_accuracy=AE_classification_testing_accuracy)
