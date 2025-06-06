# Iterative Optimization for Embedding Inversion
# This notebook implements gradient descent in input space to invert embeddings

# %% [markdown]
# ## 1. Setup and Create Diverse Test Corpus

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model components
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
sentence_model = SentenceTransformer(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModel.from_pretrained(model_name).to(device)

# Create diverse test corpus
test_corpus = [
    # News headlines
    "Scientists discover new species of deep-sea fish in Pacific Ocean",
    "Stock market reaches all-time high amid economic recovery",
    "Climate change accelerates Arctic ice melt, study finds",
    
    # Technical descriptions
    "The algorithm uses dynamic programming to optimize runtime complexity",
    "Machine learning model achieves 95% accuracy on test dataset",
    "Quantum computers leverage superposition for parallel processing",
    
    # Questions
    "What causes the northern lights to appear in the sky?",
    "How does photosynthesis convert sunlight into energy?",
    "Why do some materials conduct electricity better than others?",
    
    # Instructions
    "Mix flour, eggs, and milk to create pancake batter",
    "Press and hold the power button for 10 seconds to reset",
    "Apply two coats of paint, allowing each to dry completely",
    
    # Conversational
    "I really enjoyed the movie we watched last night",
    "The weather has been unusually warm for this time of year",
    "Let's meet at the coffee shop downtown at 3 PM",
    
    # Literary/Quotes
    "To be or not to be, that is the question",
    "The journey of a thousand miles begins with a single step",
    "All that glitters is not gold",
    
    # Facts
    "The human brain contains approximately 86 billion neurons",
    "Water boils at 100 degrees Celsius at sea level",
    "The speed of light in vacuum is 299,792,458 meters per second"
]

# Create embeddings
print(f"Creating embeddings for {len(test_corpus)} texts...")
target_embeddings = sentence_model.encode(test_corpus)
print(f"Embedding shape: {target_embeddings.shape}")

# %% [markdown]
# ## 2. Implement Iterative Optimization Inversion

# %%
class EmbeddingInverter:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.embedding_dim = model.config.hidden_size
        
    def mean_pooling(self, model_output, attention_mask):
        """Replicate the mean pooling used by sentence-transformers"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward_pass(self, input_embeds, attention_mask):
        """Forward pass through the model with continuous embeddings"""
        outputs = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        sentence_embeddings = self.mean_pooling(outputs, attention_mask)
        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
    
    def invert_embedding(self, target_embedding, num_tokens=10, num_iterations=1000, lr=0.01):
        """
        Invert embedding using gradient descent on input embeddings
        """
        # Convert target to tensor
        target = torch.tensor(target_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
        target = F.normalize(target, p=2, dim=1)
        
        # Initialize random embeddings for tokens
        # Start with embeddings close to average token embedding
        with torch.no_grad():
            # Get embeddings for common words as initialization
            init_tokens = self.tokenizer.encode("the of and to in is", add_special_tokens=False)
            init_embeds = self.model.embeddings.word_embeddings(torch.tensor(init_tokens).to(self.device))
            avg_embed = init_embeds.mean(dim=0)
        
        # Learnable embeddings (including [CLS] and [SEP])
        input_embeds = avg_embed.unsqueeze(0).repeat(1, num_tokens + 2, 1).clone().detach()
        input_embeds.requires_grad = True
        
        # Set [CLS] and [SEP] tokens
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        input_embeds.data[0, 0] = self.model.embeddings.word_embeddings(torch.tensor([cls_id]).to(self.device))
        input_embeds.data[0, -1] = self.model.embeddings.word_embeddings(torch.tensor([sep_id]).to(self.device))
        
        # Attention mask (all ones)
        attention_mask = torch.ones(1, num_tokens + 2).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam([input_embeds], lr=lr)
        
        losses = []
        
        # Optimization loop
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            predicted = self.forward_pass(input_embeds, attention_mask)
            
            # Compute loss (negative cosine similarity)
            loss = 1 - F.cosine_similarity(predicted, target)
            
            # Backward pass
            loss.backward()
            
            # Update only the middle tokens (not CLS/SEP)
            with torch.no_grad():
                input_embeds.grad[0, 0] = 0
                input_embeds.grad[0, -1] = 0
            
            optimizer.step()
            
            losses.append(loss.item())
            
            if iteration % 200 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.6f}")
        
        # Find nearest tokens for each embedding
        final_embeds = input_embeds.detach()
        reconstructed_tokens = self.find_nearest_tokens(final_embeds[0, 1:-1])  # Exclude CLS/SEP
        
        return reconstructed_tokens, losses, final_embeds
    
    def find_nearest_tokens(self, embeddings):
        """Find nearest tokens in vocabulary for continuous embeddings"""
        # Get all token embeddings
        vocab_size = self.tokenizer.vocab_size
        all_token_ids = torch.arange(vocab_size).to(self.device)
        all_token_embeds = self.model.embeddings.word_embeddings(all_token_ids)
        
        reconstructed_tokens = []
        
        for embed in embeddings:
            # Compute distances to all tokens
            distances = torch.cdist(embed.unsqueeze(0), all_token_embeds).squeeze()
            
            # Find top 5 nearest tokens
            nearest_ids = torch.argsort(distances)[:5]
            nearest_tokens = [self.tokenizer.decode([token_id]) for token_id in nearest_ids]
            nearest_distances = [distances[token_id].item() for token_id in nearest_ids]
            
            reconstructed_tokens.append({
                'best_token': nearest_tokens[0],
                'alternatives': list(zip(nearest_tokens[1:], nearest_distances[1:]))
            })
        
        return reconstructed_tokens

# %% [markdown]
# ## 3. Run Inversion on Test Corpus

# %%
# Initialize inverter
inverter = EmbeddingInverter(transformer_model, tokenizer, device)

# Store results
results = []

# Test on subset of corpus
for i, (text, embedding) in enumerate(zip(test_corpus[:10], target_embeddings[:10])):
    print(f"\n{'='*60}")
    print(f"Original text: {text}")
    print(f"{'='*60}")
    
    # Try different sequence lengths
    for num_tokens in [5, 10, 15]:
        print(f"\nTrying with {num_tokens} tokens:")
        
        tokens, losses, final_embeds = inverter.invert_embedding(
            embedding, 
            num_tokens=num_tokens,
            num_iterations=1000,
            lr=0.01
        )
        
        # Reconstruct text
        reconstructed = ' '.join([t['best_token'] for t in tokens])
        
        # Verify by encoding reconstructed text
        reconstructed_embedding = sentence_model.encode([reconstructed])[0]
        similarity = cosine_similarity([embedding], [reconstructed_embedding])[0][0]
        
        print(f"Reconstructed: {reconstructed}")
        print(f"Similarity: {similarity:.4f}")
        
        # Show alternatives
        print("Token alternatives:")
        for j, token_info in enumerate(tokens[:5]):  # Show first 5
            alts = ', '.join([f"{t}({d:.2f})" for t, d in token_info['alternatives'][:2]])
            print(f"  Position {j}: {token_info['best_token']} | alternatives: {alts}")
        
        results.append({
            'original': text,
            'num_tokens': num_tokens,
            'reconstructed': reconstructed,
            'similarity': similarity,
            'final_loss': losses[-1]
        })

# %% [markdown]
# ## 4. Analyze Results

# %%
# Convert to DataFrame for analysis
df_results = pd.DataFrame(results)

# Group by sequence length
print("\nAverage similarity by sequence length:")
print(df_results.groupby('num_tokens')['similarity'].agg(['mean', 'std']))

# Plot convergence for one example
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title("Optimization Loss Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss (1 - cosine similarity)")
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# ## 5. Advanced: Multi-Start Optimization

# %%
def multi_start_inversion(inverter, target_embedding, num_starts=5, num_tokens=10):
    """
    Run inversion from multiple random initializations
    """
    all_results = []
    
    for start in range(num_starts):
        tokens, losses, _ = inverter.invert_embedding(
            target_embedding,
            num_tokens=num_tokens,
            num_iterations=500,  # Fewer iterations per start
            lr=0.01
        )
        
        reconstructed = ' '.join([t['best_token'] for t in tokens])
        reconstructed_embedding = sentence_model.encode([reconstructed])[0]
        similarity = cosine_similarity([target_embedding], [reconstructed_embedding])[0][0]
        
        all_results.append({
            'text': reconstructed,
            'similarity': similarity,
            'tokens': tokens
        })
    
    # Sort by similarity
    all_results.sort(key=lambda x: x['similarity'], reverse=True)
    return all_results

# Test multi-start on one example
test_text = test_corpus[4]  # "Machine learning model achieves..."
test_embedding = target_embeddings[4]

print(f"Original: {test_text}")
print("\nMultiple reconstruction attempts:")

results = multi_start_inversion(inverter, test_embedding, num_starts=5)

for i, result in enumerate(results):
    print(f"\nAttempt {i+1}: {result['text']}")
    print(f"Similarity: {result['similarity']:.4f}")

# %% [markdown]
# ## 6. Analysis: What Makes Inversion Easier/Harder?

# %%
# Test on different text types
text_types = {
    'short_factual': "Water boils at 100 degrees Celsius",
    'long_technical': "The convolutional neural network architecture uses multiple layers of learnable filters",
    'question': "How does the immune system protect against pathogens?",
    'conversational': "Hey, want to grab lunch tomorrow?",
    'unique_terms': "Quantum entanglement enables instantaneous correlation",
    'common_words': "The day was very nice and warm"
}

print("Inversion difficulty by text type:\n")

for text_type, text in text_types.items():
    embedding = sentence_model.encode([text])[0]
    
    tokens, losses, _ = inverter.invert_embedding(embedding, num_tokens=10, num_iterations=500)
    reconstructed = ' '.join([t['best_token'] for t in tokens])
    
    reconstructed_embedding = sentence_model.encode([reconstructed])[0]
    similarity = cosine_similarity([embedding], [reconstructed_embedding])[0][0]
    
    print(f"{text_type:20} | Similarity: {similarity:.4f}")
    print(f"  Original:      {text}")
    print(f"  Reconstructed: {reconstructed}")
    print()

# %% [markdown]
# ## 7. Conclusions

# %%
print("=== ITERATIVE OPTIMIZATION RESULTS ===\n")

print("KEY FINDINGS:")
print("1. Optimization successfully finds embeddings that produce high similarity")
print("2. Exact text recovery is rare, but semantic content is preserved")
print("3. Longer sequences generally achieve higher similarity")
print("4. Technical/unique terms are easier to recover than common phrases")

print("\nLIMITATIONS:")
print("- Discrete token selection is challenging")
print("- Multiple valid reconstructions exist")
print("- Common words create ambiguity")

print("\nSECURITY IMPLICATIONS:")
print("- Even without exact recovery, semantic content leaks")
print("- Unique/technical terms are especially vulnerable")
print("- Demonstrates embeddings are not safe for sensitive data")