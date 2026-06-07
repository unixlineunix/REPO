import torch
from model import GPTLanguageModel

class LLMGenerator:
    def __init__(self, model_path='model.pt'):
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        self.chars = checkpoint['chars']
        self.stoi = checkpoint['stoi']
        self.itos = checkpoint['itos']
        vocab_size = checkpoint['vocab_size']

        # Determine best device (Vulkan/CUDA/CPU)
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch, 'vulkan') and torch.vulkan.is_available():
            self.device = 'vulkan'
        else:
            self.device = 'cpu'
        
        print(f"Using device: {self.device}")

        # Initialize and load model
        self.model = GPTLanguageModel(vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def generate_text(self, prompt, max_tokens=100):
        # Tokenize prompt (character-level)
        # Note: If prompt is empty, start with a newline or space
        if not prompt:
            prompt = "\n"
        
        idx = torch.tensor([self.stoi.get(c, 0) for c in prompt], dtype=torch.long, device=self.device).unsqueeze(0)
        
        # Generate
        with torch.no_grad():
            generated_idx = self.model.generate(idx, max_tokens)
            
        # Decode
        return ''.join([self.itos[i] for i in generated_idx[0].tolist()])

# Global generator instance for C API
_generator = None

def init_llm(model_path='model.pt'):
    global _generator
    _generator = LLMGenerator(model_path)
    return "LLM Initialized"

def generate(prompt, max_tokens=100):
    if _generator is None:
        return "Error: LLM not initialized"
    return _generator.generate_text(prompt, max_tokens)

if __name__ == "__main__":
    # Test if run directly
    init_llm()
    print(generate("To be", 50))
