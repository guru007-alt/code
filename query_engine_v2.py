import os
import json
import torch
import torch.nn.functional as F
from model_v2 import model, device # Imports the 8-Head Architecture
from decoder_bridge import AIDecoderBridge

class AIVectorEngine:
    def __init__(self, memory_dir='experience_memory'):
        self.memory_dir = memory_dir
        self.bridge = AIDecoderBridge()
        
        # ADD THIS LINE: Attach the global model to this class instance
        from model_v2 import model as v2_model
        self.model = v2_model 
        
        # Load the Master V2 Weights
        weight_path = 'master_brain.pth'
        if os.path.exists(weight_path):
            self.model.load_state_dict(torch.load(weight_path, weights_only=True))
            print("💎 Master Brain (V2) Aligned and Loaded.")
        
        self.model.to(device)
        self.model.eval()


    def get_live_fingerprint(self, text):
        """Processes live text into an 8-head vector fingerprint"""
        tokens = torch.tensor(list(text.encode('utf-8', errors='ignore')), dtype=torch.long).to(device)
        return model.get_fingerprint(tokens)

    def find_best_match(self, current_text):
        user_vec = self.get_live_fingerprint(current_text)
        best_match = None
        highest_sim = -1.0

        if not os.path.exists(self.memory_dir):
            return None, 0.0

        for file in os.listdir(self.memory_dir):
            if file.endswith('.exp'):
                with open(os.path.join(self.memory_dir, file), 'r') as f:
                    mem = json.load(f)
                    mem_vec = torch.tensor(mem['fingerprint']).to(device)
                
                    # Multi-head Cosine Similarity
                    sim = F.cosine_similarity(user_vec.unsqueeze(0), mem_vec.unsqueeze(0)).item()
                    
                    if sim > highest_sim:
                        highest_sim = sim
                        best_match = mem

        # Convert to a 0-100 scale
        confidence = round(highest_sim * 100, 2)
        return best_match, confidence

    def get_recall_content(self, memory, query_text=None):
        """Pulls raw data and slices it to return a clean conversation or code block"""
        category, dat_file = memory['source'].split('/')
        data = self.bridge.prepare_training_data(category, dat_file)
        text = data['decoded_text']
        
        # 1. Clean binary junk (removes non-printable characters and header remnants)
        import re
        clean_text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)

        # 2. Targeted Slicing (For Chat/Instructions)
        if query_text:
            # Split into clean lines
            lines = [l.strip() for l in clean_text.split('\n') if l.strip()]
            for i, line in enumerate(lines):
                # If your word (like 'hi') is in this line
                if query_text.lower() in line.lower():
                    # Check if there is an AI response on the next line
                    if i + 1 < len(lines):
                        ai_reply = lines[i+1]
                        # Remove the "AI:" label if it exists for a cleaner chat
                        return ai_reply.replace('AI:', '').replace('ai:', '').strip()
        
        # 3. Default Fallback (If no specific line matches, return a manageable chunk)
        return clean_text[:200].strip()

if __name__ == "__main__":
    engine = AIVectorEngine()
    # Test recall with V2 logic
    match, conf = engine.find_best_match("price calculation logic")
    if match:
        print(f"🎯 Binary Recall: {match['source']} | Confidence: {conf}%")
