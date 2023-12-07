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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# retrieve datasets
class CSVDataset(Dataset):
    def __init__(self, csv_file, class_name):
        self.data = pd.read_csv(csv_file, usecols=[0, 131], encoding='unicode_escape', header=0, delimiter=',')
        self.data = self.data.values.tolist()
        self.data = [[sublist[0]] * 2 + sublist[1:] for sublist in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the first and last elements from each row
        return self.data[idx]


# Step 2: Create a vocabulary
def create_vocab(item_tokens):
    return list(set(item_tokens))
    # set: unordered, UNIQUE elements


# Step 3: Assign an index to each bigram
def create_index_mapping(vocabulary):
    return {item: index for index, item in enumerate(vocabulary)}


# Step 4: Create one-hot encoding vectors
def word_encode(item, index_mapping):
    return index_mapping[item]


# Define the root folder where all the CSV files are located
root_folder = './DGA'

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
print(
    f"Domain Name:\n {datasets[0][2][0]}\nsecond Domain Name:\n {datasets[0][2][1]}\nClass Label encoded:\n{datasets[0][2][2]}")

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

print(
    f"Domain Bigram Name:\n {datasets[0][2][0]}\nDomain Char Name:\n {datasets[0][2][1]}\nClass Label encoded:\n{datasets[0][2][2]}")

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

# Step 4: Encode bigrams/chars vectors
for d in range(len(datasets)):  # loops the datasets
    for i in range(len(datasets[d])):  # loops domains inside a dataset
        datasets[d][i][0] = [word_encode(datasets[d][i][0][bigram], index_mapping_bigrams) for bigram in
                             range(len(datasets[d][i][0]))]
        datasets[d][i][1] = [word_encode(datasets[d][i][1][char], index_mapping_chars) for char in
                             range(len(datasets[d][i][1]))]

# Use PyTorch's ConcatDataset to concatenate the individual datasets into a single dataset
dataset_tot = ConcatDataset(datasets)
# with open("final_dataset_one_hot.txt", "w") as text_file:
print(
    f"A Dataset Domain:\n Bigram one hot:\n{dataset_tot[2][0]}\n Char encode:\n{dataset_tot[2][1]}\n Bigram encode:\n{dataset_tot[2][0]}\nLabel:\n{dataset_tot[2][2]}")
print(f'dataset len: {len(dataset_tot)}')  # 50898

del datasets
for n in range(len(dataset_tot)):
    dataset_tot[n][0] = torch.tensor(np.array(dataset_tot[n][0]), dtype=torch.long)
    dataset_tot[n][1] = torch.tensor(np.array(dataset_tot[n][1]), dtype=torch.long)
    dataset_tot[n][2] = torch.tensor(np.array(dataset_tot[n][2]), dtype=torch.long)

batch_size = 30
# train_data, test_data = random_split(dataset_tot, [48710, 12188])  # 80% - 20%, 10000-legit 1000-dataset
train_data, test_data = random_split(dataset_tot, [40710, 10188])  # 80% - 20%, 1000-dataset
# Create data loaders for dataset; shuffle for training
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)


# Define the positional encoding function
def positional_encoding(max_seq_len, d_model):
    position = torch.arange(0, max_seq_len).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # denominator
    # Create a Matrix of shape (1, seq_len, vocab_size, d_model)
    pos_enc = torch.zeros(1, max_seq_len, d_model)
    pos_enc[0, :, 0::2] = torch.sin(position * div_term)  # even
    pos_enc[0, :, 1::2] = torch.cos(position * div_term)  # odd
    print('pos enc: ', pos_enc.shape)
    return pos_enc  # (1, seq_len, d_model)


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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, conv_kernel_size, num_kernels, word_dim):
        super(BigramEmbeddingModel, self).__init__()
        self.word_dim = word_dim
        self.num_kernels = num_kernels  # 64
        self.conv_kernel_size = conv_kernel_size  # 3
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = positional_encoding(max_seq_len=self.word_dim, d_model=embedding_dim).requires_grad_(
            False).to(device)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=self.num_kernels,
                                kernel_size=self.conv_kernel_size, bias=True, padding='same')
        # self.pool = nn.MaxPool1d(kernel_size=self.conv_kernel_size-1)  # 2
        self.feedforward = nn.Sequential(
            nn.Linear(self.num_kernels, hidden_dim, bias=True),  # in: 64, out: 128
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, embedding_dim, bias=True),  # in: 128, out: 128
        )

    def forward(self, input_words):
        # print(input_words.shape)
        out = self.embedding(input_words) + self.pos_encoder
        # out -> [batch 46 128] -> batch_size, sequence_length, embedding_dim
        # print('in bigram cnn: ', out.shape)

        #  in bigram cnn:  torch.Size([30, 46, 128])
        # out = self.pool(F.relu(self.conv1d(out.permute(0, 2, 1))))
        out = F.relu(self.conv1d(out.permute(0, 2, 1))).permute(0, 2, 1)

        #  out cnn: torch.Size([10, 64, 64628])
        ##print('\nout bigram cnn: ', out.shape)

        # print('in bigram ff:', out.shape)
        # in bigram ff: torch.Size([30, 46, 64])
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
        self.dropout = nn.Dropout(p=0.1)

        # cnn
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=256,
                                kernel_size=conv_kernel_size, padding='same', bias=True)
        # kernel_size=3, stride=1, padding=1 => same convolution (the output keeps the same dimension)
        # dim input cnn = dim output attention = dim domain name sequence len
        # self.pool = nn.MaxPool1d(kernel_size=1)  # window vector length = conv kernel length
        # Spatial size after conv layer = ((in_size-kernel_size+ 2*padding)/stride)+1
        # (128 - 3 + 2 * (256 - 128 + 3 - 1) / 2) + 1 = 256
        # Define the feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(256, hidden_dim, bias=True),  # in: 256, out: 128
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, embedding_dim, bias=True),  # in: 128, out: 128
        )

        # Define the layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        ##print('input emb input dim ', input.shape)
        # -> [30 47 128] -> batch_size, sequence_length, embedding_dim

        # Multi-Head Self Attention
        # encoder: query, key, value are the same
        attn_output, _ = self.attention(input, input, input)
        ##print('attention dim: ', attn_output.shape)

        # Add and Norm
        input = self.dropout(input + attn_output)
        input = self.norm1(input)
        ##print('after norm dim: ', input.shape)

        # out = [batch_size, seq_len, embedding_dim]
        # Permute to (batch_size, embedding_dim, sequence_length)
        # print('cnn in char: ', input.shape)

        # cnn
        # cnn in char:  torch.Size([30, 47, 128])
        cnn_output = F.relu(self.conv1d(input.permute(0, 2, 1))).permute(0, 2, 1)
        # after char cnn dim:  torch.Size([30, 47, 256])
        # print('after char cnn dim: ', cnn_output.shape)

        # Feedforward
        ff_output = F.relu(self.feedforward(cnn_output))
        # feed forward char dim:  torch.Size([30, 47, 128])
        # print('feed forward char dim: ', ff_output.shape)

        # Add and Norm
        output = self.dropout(input + ff_output)
        output = self.norm2(output)
        ##print('output dim: ', output.shape)

        return output


class RepeatedTransformerEncoder(nn.Module):
    def __init__(self, vocab_size_chars, d_model, hidden_dim, num_heads, num_layers, conv_kernel_size, seq_len,
                 num_kernels_chars=256):
        super(RepeatedTransformerEncoder, self).__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size_chars, d_model)
        self.pos_encoder = positional_encoding(max_seq_len=self.seq_len, d_model=128).requires_grad_(False).to(device)
        self.layers = nn.ModuleList([
            ModifiedEncoderBlock(vocab_size_chars, d_model, hidden_dim, num_heads,
                                 conv_kernel_size=conv_kernel_size, num_kernels=num_kernels_chars,
                                 longest_word=self.seq_len)
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
        self.concat_layer = nn.Linear((longest_bigram_word + longest) * 128, num_classes, bias=True)

    def forward(self, bigram_in, char_in):
        out1 = self.model1(bigram_in)
        out2 = self.model2(char_in)
        # print(f"\nBigram out: {out1.shape}\nChar out: {out2.shape}")
        # Concatenate the outputs
        concatenated_output = torch.cat((out1, out2), dim=1)
        # print('\nconcatenated dim: ', concatenated_output.shape)
        # concatenated dim:  torch.Size([30, 11904])
        concatenated_output = concatenated_output.view(-1, 93 * 128)

        # Apply the dense layer
        # print(concatenated_output.shape)
        final_output = self.concat_layer(concatenated_output)
        ##print('\nfinal dim: ', final_output.shape)

        return final_output


# training params
learning_rate = 0.0005
num_epochs = 20
num_classes = len(labels)  # 51

bigram_model = BigramEmbeddingModel(vocab_size=vocab_size_bigrams, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                    conv_kernel_size=conv_kernel_size, num_kernels=num_kernels_bigrams,
                                    word_dim=longest_bigram_word).to(device)
char_model = RepeatedTransformerEncoder(vocab_size_chars=vocab_size_chars, d_model=embedding_dim, hidden_dim=hidden_dim,
                                        num_heads=num_heads, num_layers=num_layers, conv_kernel_size=conv_kernel_size,
                                        num_kernels_chars=num_kernels_chars, seq_len=longest).to(device)
model = DGAHybridModel(bigram_model, char_model, num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
n_total_steps = len(train_loader)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for step, (input_bigrams, input_chars, target) in enumerate(train_loader):

        input_bigrams = input_bigrams.to(device)
        # 30, 46, => bigram input shape
        ##print(f'input bigrams: {input_bigrams.shape}')

        input_chars = input_chars.to(device)
        # 30, 47 => char input shape
        ##print(f'input chars: {input_chars.shape}')

        target = target.to(device)

        # forward
        model_out = model(input_bigrams, input_chars)
        ##print('output model: ', model_out.shape)

        loss = criterion(model_out, target)
        # backward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(
                f'epoch {epoch + 1} / {num_epochs}, step {step + 1}/{n_total_steps}, loss = {loss.item():.4f}')

# testing and evaluation
with torch.no_grad():
    model.eval()
    n_true_positive = 0  # true positive
    n_samples = 0
    class_support = [0 for i in range(51)]  # n samples per class
    n_false_negative = 0
    n_false_positive = 0
    label_idx = {i: label for i, label in enumerate(labels)}
    n_class_false_positive = [0 for i in range(51)]
    n_class_false_negative = [0 for i in range(51)]  # false negative for each class
    n_class_true_positive = [0 for i in range(51)]  # true positive for each class
    n_class_samples = [0 for i in range(51)]  # n. samples for each class
    for input_bigrams, input_chars, classes in test_loader:
        input_bigrams = input_bigrams.to(device)
        input_chars = input_chars.to(device)
        classes = classes.to(device)
        outputs = model(input_bigrams, input_chars)

        # value, index
        _, predicted = torch.max(outputs, 1)  # we don't need the actual value, just the class label (predictions)
        n_true_positive = (predicted == classes).sum().item()  # for each correct prediction we will add 1

        for i in range(batch_size):
            actual = classes[i]
            pred = predicted[i]
            class_support[actual] += 1
            if actual == pred:
                n_class_true_positive[actual] += 1
            else:
                n_class_false_negative[actual] += 1
                n_class_false_positive[pred] += 1
            n_class_samples[actual] += 1

    macro_avg_precision = 0
    macro_avg_recall = 0
    # precision, recall, F1 for each single class
    for i in range(51):
        # individual class
        class_precision = 0.0000
        class_recall = 0.0000
        class_f1 = 0.0000
        if class_support[i] != 0 and n_class_true_positive[i] != 0:
            class_precision = n_class_true_positive[i] / (n_class_true_positive[i] + n_class_false_positive[i])
            class_recall = n_class_true_positive[i] / (n_class_true_positive[i] + n_class_false_negative[i])
            class_f1 = (2 * class_precision * class_recall) / (class_precision + class_recall)

        # add contribution from that class to entire model
        macro_avg_precision += class_precision
        macro_avg_recall += class_recall
        n_false_negative += n_class_false_negative[i]
        n_false_positive += n_class_false_positive[i]

        print(
            f'{i}. Class: {label_idx[i]} - Support: {class_support[i]}  ->  Precision: {class_precision:.4f}  Recall: {class_recall:.4f}  F1: {class_f1:.4f}  False negatives: {n_class_false_negative[i]}  True positives: {n_class_true_positive[i]}')

    # micro_precision = n_true_positive / (n_true_positive + n_false_positive)
    # micro_recall = n_true_positive / (n_true_positive + n_false_negative)
    # micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)

    macro_avg_precision = macro_avg_precision / 51
    macro_avg_recall = macro_avg_recall / 51
    macro_avg_f1 = (2 * macro_avg_precision * macro_avg_recall) / (macro_avg_precision + macro_avg_recall)

    # multiclass : tot fp = tot fn -> precision=recall
    print(
        #f'\nmodel micro avg precision = {micro_precision:.4f}\nmodel micro avg recall = {micro_recall:.4f}\nmodel micro avg F1 = {micro_f1:.4f}'
        f'\nmodel MACRO avg precision = {macro_avg_precision:.4f}\nmodel MACRO avg recall = {macro_avg_recall:.4f}\nmodel MACRO avg F1 = {macro_avg_f1:.4f}\n')