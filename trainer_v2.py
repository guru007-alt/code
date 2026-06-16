import os
import torch
import json
import random
import torch.optim as optim
from decoder_bridge import AIDecoderBridge
from model_v2 import model, device, block_size

class MasterTrainer:
    def __init__(self):
        self.bridge = AIDecoderBridge()
        self.memory_dir = 'experience_memory'
        self.backup_dir = 'backups'
        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    def train_session(self, category, filename, steps=300):
        data = self.bridge.prepare_training_data(category, filename)
        if not data: return

        print(f"🧬 Master Alignment: {filename} ({category})")
        
        # Convert to bytes
        text_bytes = data['decoded_text'].encode('utf-8', errors='ignore')
        tokens = torch.tensor(list(text_bytes), dtype=torch.long).to(device)

        # Optimization settings
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        model.train()
        
        for i in range(steps):
            # Random sampling for robust learning
            if len(tokens) <= block_size:
                x = tokens[:-1].unsqueeze(0)
                y = tokens[1:].unsqueeze(0)
            else:
                start = random.randint(0, len(tokens) - block_size - 1)
                chunk = tokens[start : start + block_size + 1]
                x = chunk[:-1].unsqueeze(0)
                y = chunk[1:].unsqueeze(0)
            
            # Forward pass
            logits, loss = model(x, y)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if i % 50 == 0:
                print(f"   📊 Step {i}/{steps} | Neural Loss: {loss.item():.4f}")

        # 1. Generate the Master Fingerprint
        model.eval()
        with torch.no_grad():
            fingerprint = model.get_fingerprint(tokens)
        
        # 2. Save Experience
        self.save_experience(category, filename, fingerprint)
        
        # 3. Autonomous Backup System
        weight_path = 'master_brain.pth'
        if os.path.exists(weight_path):
            # Save old version to backups before overwriting
            backup_path = os.path.join(self.backup_dir, f"brain_v{len(os.listdir(self.backup_dir))}.pth")
            os.replace(weight_path, backup_path)
            
        torch.save(model.state_dict(), weight_path)
        print(f"💾 Evolution Complete. Weights backed up and updated.")

    def save_experience(self, category, filename, fingerprint):
        exp_path = os.path.join(self.memory_dir, filename.replace('.dat', '.exp'))
        experience = {
            "source": f"{category}/{filename}",
            "category": category,
            "fingerprint": fingerprint.tolist(),
            "version": "V2-8Head"
        }
        with open(exp_path, 'w') as f:
            json.dump(experience, f, indent=4)

if __name__ == "__main__":
    agent = MasterTrainer()
    # Initial alignment on your Python binary
    agent.train_session('Python', 'python_data_1.dat', steps=500)
    # Alignment on English context
    # agent.train_session('English', 'english_data_1.dat', steps=500)
