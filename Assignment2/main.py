import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerEncoder, FeedforwardClassifier, SpeechSegmentModel
from transformer import SpeechSegmentModel, TransformerDecoder
from utilities import Utilities

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy

def compute_perplexity(decoderLMmodel, data_loader, criterion, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = decoderLMmodel(X) # your model should be computing the cross entropy loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), Y.view(-1))
            losses.append(loss.item())
            if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
  
    inputfile = "speechesdataset/train_LM.txt"
    inputfile_obama = "speechesdataset/test_LM_obama.txt"
    inputfile_wbush = "speechesdataset/test_LM_wbush.txt"
    inputfile_hbush = "speechesdataset/test_LM_hbush.txt"
    
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()             
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
    
    with open(inputfile_obama, 'r', encoding='utf-8') as f:
        obamatestText = f.read()    
    test_obama_dataset = LanguageModelingDataset(tokenizer, obamatestText,  block_size)
    test_obama_loader = DataLoader(test_obama_dataset, batch_size=batch_size, shuffle=True)
    
    with open(inputfile_wbush, 'r', encoding='utf-8') as f:
        wbushtestText = f.read()    
    test_wbush_dataset = LanguageModelingDataset(tokenizer, wbushtestText,  block_size)
    test_wbush_loader = DataLoader(test_wbush_dataset, batch_size=batch_size, shuffle=True)
    
    with open(inputfile_hbush, 'r', encoding='utf-8') as f:
        hbushtestText = f.read()     
    test_hbush_dataset = LanguageModelingDataset(tokenizer, hbushtestText,  block_size)
    test_hbush_loader = DataLoader(test_hbush_dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    encoder = TransformerEncoder(n_embd=n_embd, n_head=n_head, n_layer=n_layer, vocab_size=tokenizer.vocab_size)
    classifier = FeedforwardClassifier(n_input=n_input, n_hidden=n_hidden, n_output=n_output)
    encoder_model = SpeechSegmentModel(encoder, classifier).to(device)
    
    num_encoder_parameters = sum(p.numel() for p in encoder_model.parameters())
    print(f"Total trainable parameters for encoder_model: {num_encoder_parameters}")
    # Optimizer and loss function
    optimizer = optim.Adam(encoder_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    encoder_model.train()
     # for the classification task, you will train for a fixed number of epochs like this:
    for epoch in range(epochs_CLS):
        total_loss = 0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            # CLS training code here
            optimizer.zero_grad()           # Reset gradients to zero for each batch
            output = encoder_model(xb)      # Forward pass
            loss = criterion(output, yb)    # Compute loss
            loss.backward()                 # Backpropagate the loss
            optimizer.step()                # Update the model parameters
            total_loss += loss.item()

        # Calculate average loss and test accuracy for the epoch
        average_loss = total_loss / len(train_CLS_loader)
        test_accuracy = compute_classifier_accuracy(encoder_model, test_CLS_loader)
        print(f'Epoch {epoch + 1}, Loss: {average_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
    encoder_utils = Utilities(tokenizer, encoder_model)
    sentence = "When one nation pursues a nuclear weapon, the risk of nuclear attack rises for all nations."
    encoder_utils.sanity_check(sentence, block_size, task='Encoder')

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    decoder_model = TransformerDecoder(n_layer, n_embd, n_head, tokenizer.vocab_size).to(device)
    num_decoder_parameters = sum(p.numel() for p in decoder_model.parameters())
    print(f"Total trainable parameters for decoder_model: {num_decoder_parameters}")
    optimizer = optim.Adam(decoder_model.parameters(), lr=learning_rate)
    
    decoder_model.train()
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        # LM training code here
        optimizer.zero_grad()           # Reset gradients to zero for each batch
        outputs = decoder_model(xb)      # Forward pass
        outputs = outputs.view(-1, outputs.size(-1))
        yb = yb.view(-1)
        loss = criterion(outputs, yb)    # Compute loss
        loss.backward()                 # Backpropagate the loss
        optimizer.step()                # Update the model parameters
        total_loss += loss.item()
    
        if (i + 1) % 100 == 0:
            print(f"Step {i + 1} Train Perplexity: {compute_perplexity(decoder_model, train_LM_loader, criterion)}")
            
    print(f"LM Training Loss: {total_loss / max_iters}")

    decoder_utils = Utilities(tokenizer, decoder_model)
    decoder_utils.sanity_check(sentence, block_size, task='Decoder')
    
    print(f"Step 500 Obama Perplexity: {compute_perplexity(decoder_model, test_obama_loader, criterion)}")
    print(f"Step 500 H. Bush Perplexity: {compute_perplexity(decoder_model, test_hbush_loader, criterion)}")
    print(f"Step 500 W. Bush Perplexity: {compute_perplexity(decoder_model, test_wbush_loader, criterion)}")

if __name__ == "__main__":
    main()
