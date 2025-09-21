import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import time
from collections import defaultdict
import matplotlib.pyplot as plt

class TextDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=20):
        print("Loading WikiText-2 dataset...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        try:
            self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Attempting alternative loading method...")
            self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir='./cache')
        
        self.samples = []
        for item in self.dataset['train']:
            text = item['text']
            if text.strip():
                tokens = self.tokenizer.encode(text)
                if len(tokens) >= seq_length:
                    for i in range(0, len(tokens) - seq_length + 1, seq_length):
                        self.samples.append(tokens[i:i + seq_length])
                        if len(self.samples) >= num_samples:
                            print(f"Collected {len(self.samples)} samples")
                            return
        print(f"Collected {len(self.samples)} samples")
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx])

class RepeatedTextDataset(Dataset):
    def __init__(self, num_samples=1000, seq_length=20):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Create sample text (alternatively, load from a local file)
        text = """The quick brown fox jumps over the lazy dog. 
                 To be or not to be, that is the question.
                 In the beginning there was code.""" * 500  # Repeat to get enough samples
        
        self.samples = []
        tokens = self.tokenizer.encode(text)
        for i in range(0, len(tokens) - seq_length, seq_length):
            self.samples.append(tokens[i:i + seq_length])
            if len(self.samples) >= num_samples:
                break
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx])


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, l1_lambda=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.l1_lambda = l1_lambda
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
        
    def loss_function(self, x, x_hat, encoded):
        reconstruction_loss = nn.MSELoss()(x_hat, x)
        l1_loss = self.l1_lambda * encoded.abs().mean()
        return reconstruction_loss + l1_loss

def collect_activations(model, dataset, layer_idx=6):
    print("Collecting activations...")
    start_time = time.time()
    activations = []
    
    def hook(module, input, output):
        activations.append(output[0].detach())
    
    hook_handle = model.transformer.h[layer_idx].register_forward_hook(hook)
    
    with torch.no_grad():
        for idx, batch in enumerate(DataLoader(dataset, batch_size=32)):
            if idx % 10 == 0:
                print(f"Processing batch {idx}...")
            model(batch)
            if len(activations) * 32 >= len(dataset):
                break
    
    hook_handle.remove()
    print(f"Activation collection took {time.time() - start_time:.2f} seconds")
    return torch.cat(activations, dim=0)

def train_autoencoder(autoencoder, activations, num_epochs=100):
    print("Training autoencoder...")
    start_time = time.time()
    loader = DataLoader(TensorDataset(activations), batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            x_hat, encoded = autoencoder(x)
            loss = autoencoder.loss_function(x, x_hat, encoded)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    print(f"Training took {time.time() - start_time:.2f} seconds")
    return losses

def interpret_features(autoencoder, model, tokenizer, dataset, num_top_examples=10):
    print("Interpreting features...")
    all_activations = []
    all_tokens = []
    encoded_features = []
    
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=32):
            tokens = batch
            outputs = model(tokens, output_hidden_states=True)
            layer_activations = outputs.hidden_states[6]
            
            encoded, _ = autoencoder.encoder(layer_activations)
            
            all_activations.append(layer_activations)
            all_tokens.append(tokens)
            encoded_features.append(encoded)
    
    all_activations = torch.cat(all_activations)
    all_tokens = torch.cat(all_tokens)
    encoded_features = torch.cat(encoded_features)
    
    feature_examples = defaultdict(list)
    
    for feature_idx in range(encoded_features.shape[1]):
        if feature_idx % 10 == 0:
            print(f"Processing feature {feature_idx}...")
            
        feature_activations = encoded_features[:, feature_idx]
        top_indices = torch.argsort(feature_activations, descending=True)[:num_top_examples]
        
        for idx in top_indices:
            token_idx = idx // tokens.shape[1]
            pos_idx = idx % tokens.shape[1]
            
            start = max(0, pos_idx - 5)
            end = min(tokens.shape[1], pos_idx + 6)
            context = all_tokens[token_idx, start:end]
            
            text = tokenizer.decode(context)
            activation_value = feature_activations[idx].item()
            
            feature_examples[feature_idx].append((text, activation_value))
    
    return feature_examples

def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()

def main():
    print("Initializing...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    hidden_dim = model.config.hidden_size
    
    dataset = RepeatedTextDatasetTextDataset(num_samples=1000)
    activations = collect_activations(model, dataset)
    
    autoencoder = SparseAutoencoder(hidden_dim, hidden_dim*2)
    losses = train_autoencoder(autoencoder, activations)
    plot_losses(losses)
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    feature_examples = interpret_features(autoencoder, model, tokenizer, dataset)
    
    print("\nTop activating examples for first 5 features:")
    for feature_idx in range(5):
        print(f"\nFeature {feature_idx}:")
        for text, value in feature_examples[feature_idx][:3]:
            print(f"  {text} (activation: {value:.3f})")
    
    torch.save(autoencoder.state_dict(), 'sparse_autoencoder.pt')
    print("\nModel saved to 'sparse_autoencoder.pt'")
    print("Loss plot saved to 'training_loss.png'")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")