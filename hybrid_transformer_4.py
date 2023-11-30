import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import math
from torch.utils.data import random_split

# device config
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# retrieve datasets
class CSVDataset(Dataset):
    def __init__(self, csv_file, class_name):
        self.data = pd.read_csv(csv_file, usecols=[0, 131], encoding='unicode_escape', header=0, delimiter=',')
        self.data = self.data.values.tolist()
        self.data = [[sublist[0]] * 2 + sublist[1:] for sublist in self.data]
        self.class_name = class_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the first and last elements from each row
        return self.data[idx][0], self.class_name


# Step 2: Create a vocabulary
def create_vocab(item_tokens):
    return list(set(item_tokens))
    # set: unordered, UNIQUE elements


# Step 3: Assign an index to each bigram
def create_index_mapping(vocabulary):
    return {item: index for index, item in enumerate(vocabulary)}


# Step 4: Create one-hot encoding vectors
def one_hot_encode(item, index_mapping):
    vector = np.zeros(len(index_mapping))  # zeros vector dim = all different kinds of bigrams/chars
    vector[index_mapping[item]] = 1
    return vector


# Define the root folder where all the CSV files are located
root_folder = '/Users/massimomecarelli/Documents/HybridTransformerDGA/DGA'

# Use the os module to navigate through the directories and locate all the CSV files
csv_files = []
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith('.csv'):
            csv_files.append(os.path.join(dirpath, filename))


# Instantiate the custom dataset class for each CSV file
datasets = []
longest = 0  # longest string dim
longest_string = []
for csv_file in csv_files:
    class_name = os.path.basename(os.path.dirname(csv_file))
    dataset = CSVDataset(csv_file, class_name)
    datasets.append(dataset.data)
    # find the longest domain name
    for i in range(len(dataset)):
        longest_string = max(dataset[i][0], longest_string, key=len)
    longest = len(longest_string)
print('longest string:', longest)
print('First dataset:\n', datasets[0])

# Get unique labels and create a mapping
labels = list(set({datasets[d][0][2] for d in range(len(datasets))}))
label_to_index = {label: i for i, label in enumerate(labels)}
print(f'Labels:\n{labels}')
print(f'Labels to idx:\n{label_to_index}')

# insert a zeros padding as the header of the domain shorter than the longest
for d in range(len(datasets)):
    for i in range(len(datasets[d])):
        diff = longest - len(datasets[d][i][0])
        zero_head = ''
        for f in range(diff):
            zero_head = zero_head + '0'
        datasets[d][i][0] = zero_head + datasets[d][i][0]
        datasets[d][i][1] = zero_head + datasets[d][i][1]
        # Class labels encoded into numeric value
        datasets[d][i][2] = label_to_index[datasets[d][i][2]]

        # with open("head_dataset.txt", "w") as text_file:
print(f"Domain Name:\n {datasets[0][2][0]}\nsecond Domain Name:\n {datasets[0][2][1]}\nClass Label encoded:\n{datasets[0][2][2]}")

# ---------------------- data preparation ---------------------- #

# bigram dtaset
bigrams_ = []
# Step 1: Transform domain names in bigrams in each dataset
for d in range(len(datasets)):
    for i in range(len(datasets[d])):
        # Tokenize text in 2-grams
        tokens = list(datasets[d][i][0])
        datasets[d][i][0] = [''.join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
        bigrams_.extend(datasets[d][i][0])


# character dataset
chars_ = []
# Step 1: Transform domain names in char sequences in each dataset
for d in range(len(datasets)):
    for i in range(len(datasets[d])):
        # Tokenize text in character sequences
        datasets[d][i][1] = list(datasets[d][i][1])
        chars_.extend(datasets[d][i][1])

print(f"Domain Bigram Name:\n {datasets[0][2][0]}\nDomain Char Name:\n {datasets[0][2][1]}\nClass Label encoded:\n{datasets[0][2][2]}")


# Step 2: Create a vocabulary
bigrams_vocabulary = create_vocab(bigrams_)
chars_vocabulary = create_vocab(chars_)

print('Characters Vocabulary:')
print(chars_vocabulary)
print(f'Char Vocabulary len: {len(chars_vocabulary)}')
print('Bigrams Vocabulary:')
print(bigrams_vocabulary)
print(f'Vocabulary len: {len(bigrams_vocabulary)}')


# Step 3: Assign an index to each bigram/character
index_mapping_bigrams = create_index_mapping(bigrams_vocabulary)
index_mapping_chars = create_index_mapping(chars_vocabulary)


# Step 4: Create one-hot encoded vectors
for d in range(len(datasets)):  # loops the datasets
    for i in range(len(datasets[d])):  # loops domains inside a dataset
        datasets[d][i][0] = [one_hot_encode(datasets[d][i][0][bigram], index_mapping_bigrams) for bigram in
                             range(len(datasets[d][i][0]))]
        datasets[d][i][1] = [one_hot_encode(datasets[d][i][1][char], index_mapping_chars) for char in
                             range(len(datasets[d][i][1]))]


# Use PyTorch's ConcatDataset to concatenate the individual datasets into a single dataset
dataset_tot = ConcatDataset(datasets)
#with open("final_dataset_one_hot.txt", "w") as text_file:
print(f"A Dataset Domain:\n Bigram one hot:\n{dataset_tot[2][0]}\n Char one hot:\n{dataset_tot[2][1]}\nLabel:\n{dataset_tot[2][2]}")
print(f'dataset len: {len(dataset_tot)}')  # 50898

del datasets
for n in range(len(dataset_tot)):
    dataset_tot[n][0] = torch.tensor(np.array(dataset_tot[n][0]), dtype=torch.long)
    dataset_tot[n][1] = torch.tensor(np.array(dataset_tot[n][1]), dtype=torch.long)
    dataset_tot[n][2] = torch.tensor(np.array(dataset_tot[n][2]), dtype=torch.long)

batch_size = 2
train_data, test_data = random_split(dataset_tot, [40710, 10188])  # 80% - 20%
# Create data loaders for dataset; shuffle for training
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)


# Define the positional encoding function
def positional_encoding(max_seq_len, d_model):
    position = torch.arange(0, max_seq_len).unsqueeze(1) # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) # denominator
    # Create a Matrix of shape (1, seq_len, d_model)
    pos_enc = torch.zeros(1, max_seq_len, d_model)
    pos_enc[0, :, 0::2] = torch.sin(position * div_term) # even
    pos_enc[0, :, 1::2] = torch.cos(position * div_term) # odd
    print('pos enc: ', pos_enc.shape)
    return pos_enc # (1, seq_len, d_model)


# Models hyperparameters
embedding_dim = 128  # d_model
hidden_dim = 128  # Hidden dimension

conv_kernel_size = 3
num_kernels_bigrams = 64
longest_bigram_word = longest - 1
vocab_size_bigrams = len(bigrams_vocabulary)

vocab_size_chars = len(chars_vocabulary)
num_heads = 8  # Number of attention heads
num_layers = 3  # Window size for local attention
num_kernels_chars = 256


# 2-gram model
class BigramEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, conv_kernel_size, num_kernels, longest_word):
        super(BigramEmbeddingModel, self).__init__()
        print(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.longest_word = longest_word
        self.num_kernels = num_kernels
        self.conv_kernel_size = conv_kernel_size # 3
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = positional_encoding(max_seq_len=(vocab_size*longest_word), d_model=embedding_dim).requires_grad_(False)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=self.num_kernels, kernel_size=self.conv_kernel_size, bias=True)
        self.pool = nn.MaxPool1d(kernel_size=self.conv_kernel_size-1)  # window vector length = conv kernel length
        self.feedforward = nn.Sequential(
            nn.Linear(self.num_kernels, hidden_dim, bias=True),  # in: 64, out: 128
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim, bias=True),  # in: 128, out: 128
        )

    def forward(self, input_words):
        out = self.embedding(input_words) + self.pos_encoder
        # out -> [batch 64630 128] -> batch_size, sequence_length, embedding_dim
        # Permute to (batch_size, embedding_dim, sequence_length)
        out = out.permute(0, 2, 1)
        ##print('in cnn: ', out.shape)
        #  in cnn:  torch.Size([10, 128, 64630])

        out = self.pool(F.relu(self.conv1d(out))).permute(0, 2, 1)

        #out = F.relu(self.conv1d(out)).permute(0, 2, 1)
        #  out cnn: torch.Size([10, 64, 64628])
        ##print('\nout bigram cnn: ', out.shape)

        #out = out.view(-1, self.num_kernels*self.conv_kernel_size)
        #out = out.view(-1, self.num_kernels*((self.vocab_size * self.longest_word) - self.conv_kernel_size + 1))  # -1, 64*64628
        #print('in ff:', out.shape)

        out = self.feedforward(out)
        return out


# character model

class ModifiedEncoderBlock(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads, conv_kernel_size, longest_word,
                 num_kernels=256):
        super(ModifiedEncoderBlock, self).__init__()

        # Define the multi-head self-attention layer
        # embedding_dim=128 , heads=8 => d_k=d_v=16
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads)

        # cnn
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=256,
                                kernel_size=conv_kernel_size, padding=1, bias=True)
        # kernel_size=3, stride=1, padding=1 => same convolution (the output keeps the same dimension)
        # input cnn = output attention = domain name sequence len
        self.pool = nn.MaxPool1d(
            kernel_size=1)  # window vector length = conv kernel length
        # Spatial size after conv layer = ((in_size-kernel_size+ 2*padding)/stride)+1
        # (128 - 3 + 2 * (256 - 128 + 3 - 1) / 2) + 1 = 256
        # Define the feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(256, hidden_dim, bias=True),  # in: 38, out: 128
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim, bias=True),  # in: 128, out: 128
        )

        # Define the layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        ##print('input emb input dim ', input.shape)
        # -> [10 1786 128] -> batch_size, sequence_length, embedding_dim

        # Multi-Head Self Attention
        # encoder: query, key, value are the same
        attn_output, _ = self.attention(input, input, input)
        ##print('attention dim: ', attn_output.shape)

        # Add and Norm
        input = input + attn_output
        input = self.norm1(input)
        ##print('after norm dim: ', input.shape)

        # out = [batch_size, seq_len, embedding_dim]
        # Permute to (batch_size, embedding_dim, sequence_length)
        input = input.permute(0, 2, 1)
        ##print('out dim: ', input.shape)
        # cnn
        # out = self.pool(F.relu(self.conv1d(out))).squeeze(dim=2).permute(0, 2, 1)
        cnn_output = self.pool(F.relu(self.conv1d(input))).permute(0, 2, 1)
        #cnn_output = F.relu(self.conv1d(input)).permute(0, 2, 1)
        input = input.permute(0, 2, 1)
        ##print('after cnn dim: ', cnn_output.shape)

        # Feedforward
        ff_output = F.relu(self.feedforward(cnn_output))
        ##print('feed forward dim: ', ff_output.shape)

        # Add and Norm
        output = input + ff_output
        output = self.norm2(output)
        ##print('output dim: ', output.shape)

        return ff_output


class RepeatedTransformerEncoder(nn.Module):
    def __init__(self, vocab_size_chars, d_model, hidden_dim, num_heads, num_layers, conv_kernel_size, longest_word,
                 num_kernels_chars=256):
        super(RepeatedTransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size_chars, d_model)
        self.pos_encoder = positional_encoding(max_seq_len=1786, d_model=128).requires_grad_(False)
        self.layers = nn.ModuleList([
            ModifiedEncoderBlock(vocab_size_chars, d_model, hidden_dim, num_heads,
                                 conv_kernel_size=conv_kernel_size, num_kernels=num_kernels_chars,
                                 longest_word=longest_word)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        for layer in self.layers:
            x = layer(x)
        return x


class DGAHybridModel(nn.Module):
    def __init__(self, bigram_model, char_model, num_classes):
        super(DGAHybridModel, self).__init__()
        self.model1 = bigram_model
        self.model2 = char_model
        self.concat_layer = nn.Linear(((46*1405-1)//2)*128+47*38*128, num_classes) # 8500992

    def forward(self, bigram_in, char_in):
        out1 = self.model1(bigram_in)
        out2 = self.model2(char_in)
        ##print(f"\nBigram out: {out1.shape}\nChar out: {out2.shape}")
        # Concatenate the outputs
        concatenated_output = torch.cat((out1.view(-1, out1.shape[1]*out1.shape[2]),
                                         out2.view(-1, out2.shape[1]*out2.shape[2])), dim=1)
        ##print('\nconcatenated dim: ', concatenated_output.shape)
        # Apply the dense layer
        #final_output = F.softmax(self.concat_layer(concatenated_output), dim=1)
        final_output = self.concat_layer(concatenated_output)
        ##print('\nfinal dim: ', final_output.shape)
        #final_output = final_output.view(-1, 66414*51)

        return final_output


# training params
learning_rate = 0.001
num_epochs = 2
num_classes = len(labels)  # 51

bigram_model = BigramEmbeddingModel(vocab_size=vocab_size_bigrams, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                    conv_kernel_size=conv_kernel_size, num_kernels=num_kernels_bigrams,
                                    longest_word=longest_bigram_word)
char_model = RepeatedTransformerEncoder(vocab_size_chars=vocab_size_chars, d_model=embedding_dim, hidden_dim=hidden_dim,
                                        num_heads=num_heads, num_layers=num_layers, conv_kernel_size=conv_kernel_size,
                                        num_kernels_chars=num_kernels_chars, longest_word=longest)
model = DGAHybridModel(bigram_model, char_model, num_classes)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
n_total_steps = len(train_loader)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for step, (input_bigrams, input_chars, label) in enumerate(train_loader):

        # 10, 46, 1405 => bigram input shape
        # 10 64630 => reshape (flatten the last two dim)
        ##print(f'input bigrams: {input_bigrams.shape}')
        input_bigrams = input_bigrams.view(-1, 46 * 1405)

        # 10, 47, 38 => char input shape
        # 10, 1786 => reshape
        ##print(f'input chars: {input_chars.shape}')
        input_chars = input_chars.view(-1, 47 * 38)

        # forward
        model_out = model(input_bigrams, input_chars)
        ##print('output model: ', model_out.shape)

        loss = criterion(model_out, label)
        # backward
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(
                f'epoch {epoch + 1} / {num_epochs}, step {step + 1}/{n_total_steps}, loss = {loss.item():.4f}')

# testing and evaluation
with torch.no_grad():
    model.eval()
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(51)]
    n_class_samples = [0 for i in range(51)]
    for input_bigrams, input_chars, classes in test_loader:
        input_bigrams = input_bigrams.reshape(-1, 46 * 1405)
        input_chars = input_chars.view(-1, 47 * 38)
        # labels = labels.to(device)
        outputs = model(input_bigrams, input_chars)

        # value, index
        _, predicted = torch.max(outputs, 1)  # we don't need the actual value, just the class label (predictions)
        n_samples += classes.shape[0]  # number of samples in the current batch
        n_correct = (predicted == classes).sum().item()  # for each correct prediction we will add 1

        for i in range(batch_size):
            label = label_to_index[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')

    # accuracy for each single class
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {label_to_index[i]}: {acc} %')
