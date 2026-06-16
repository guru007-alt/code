import os
import json
import torch
import torch.nn.functional as F
import subprocess
from query_engine_v2 import AIVectorEngine
from model_v2 import device, block_size

class AutonomousAgent:
    def __init__(self):
        self.engine = AIVectorEngine()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.laws_path = os.path.join(self.base_dir, 'templates', 'system_laws.txt')

    def read_laws(self):
        if os.path.exists(self.laws_path):
            with open(self.laws_path, 'r', encoding='utf-8') as f:
                return f.read()
        return "No laws established."

    def observe_environment(self):
        memories = os.listdir('experience_memory') if os.path.exists('experience_memory') else []
        backups = os.listdir('backups') if os.path.exists('backups') else []
        return {
            "memory_count": len(memories),
            "version_count": len(backups),
            "gpu_active": device == 'cuda'
        }

    def generate_thought(self, prompt, max_new_tokens=100, temperature=0.7):
        """GENERATIVE ENGINE: Predicts the next bytes based on binary ideology"""
        self.engine.model.eval()
        
        # 1. Convert prompt to tokens (bytes)
        idx = torch.tensor(list(prompt.encode('utf-8')), dtype=torch.long).to(device).unsqueeze(0)
        
        # 2. Generative Loop
        generated_tokens = []
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx[:, -block_size:]
            
            # Get predictions
            with torch.no_grad():
                logits, _ = self.engine.model(idx_cond)
                logits = logits[:, -1, :] / temperature # Apply creativity
                probs = F.softmax(logits, dim=-1)
                
                # Sample the next byte
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
                
                # Stop if we hit a newline or special end (optional)
                token_val = next_token.item()
                generated_tokens.append(token_val)
                if token_val == 10: # Stop at newline
                    break
        
        try:
            return bytes(generated_tokens).decode('utf-8', errors='ignore')
        except:
            return "[Error Decoding Binary Thought]"

    def reason_and_act(self, user_input):
        laws = self.read_laws()
        env = self.observe_environment()
        
        print(f"📡 System Pulse: Versions={env['version_count']} | Memory={env['memory_count']}")

        # ACTION: Self-Modification Logic
        if "upgrade" in user_input.lower() or "learn" in user_input.lower():
            if env['gpu_active']:
                venv_python = os.path.join(self.base_dir, 'venv_brain', 'Scripts', 'python.exe')
                if not os.path.exists(venv_python):
                    venv_python = os.path.join(self.base_dir, 'venv', 'Scripts', 'python.exe')
                
                subprocess.run([venv_python, "trainer_v2.py"])
                return "🛠️ [EVOLUTION] Self-modification complete. My binary neural weights have been upgraded."
            return "⚠️ Cannot evolve. GPU acceleration is unavailable."

        # GENERATION: Instead of finding a match, we GENERATE a response
        # We find a match to use as 'contextual inspiration' for the generation
        match, score = self.engine.find_best_match(user_input)
        
        context_prompt = f"System Laws: {laws}\nUser: {user_input}\nContext: "
        if match and score > 30.0:
            context_prompt += self.engine.get_recall_content(match)[:150]
        
        # The AI 'Thinks' and generates a new reply
        print("🧠 Generative Inference Active...")
        reply = self.generate_thought(context_prompt)
        
        return f"✨ [GENERATED] {reply.strip()}"

if __name__ == "__main__":
    agent = AutonomousAgent()
    print(agent.reason_and_act("hi"))
