from flask import Flask, render_template, request, jsonify
import os
import torch
from trainer_v2 import MasterTrainer
from autonomous_agent import AutonomousAgent

app = Flask(__name__)
# Fixes the emoji glitch (ðŸ§ )
app.config['JSON_AS_ASCII'] = False 

# Initialize Brain components once
trainer = MasterTrainer()
agent = AutonomousAgent()

KNOWLEDGE_DIR = 'Knowledge_Base'
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Use force=True to ensure JSON is captured even if headers are wonky
    data = request.get_json(force=True)
    user_msg = data.get('message', '')
    
    # The Agent observes, reasons, and acts
    response = agent.reason_and_act(user_msg)
    return jsonify({'reply': response})

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'msg': 'No file part'})
    
    file = request.files['file']
    category = request.form.get('category', 'General')
    
    if file and file.filename != '':
        cat_dir = os.path.join(KNOWLEDGE_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        save_path = os.path.join(cat_dir, file.filename)
        file.save(save_path)
        return jsonify({'status': 'success', 'msg': f'Uploaded to {category}'})
    
    return jsonify({'status': 'error', 'msg': 'Invalid file upload'})

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json(force=True)
    category = data.get('category')
    filename = data.get('filename')
    
    print(f"⚡ Neural Alignment Initiated: {filename} in {category}")
    
    # 1. Trigger the V2 Master Trainer
    trainer.train_session(category, filename, steps=400)
    
    # 2. REFRESH: Force the agent to reload the new brain weights
    agent.engine.__init__() 
    
    return jsonify({'status': 'success', 'msg': 'Evolution Complete'})

@app.route('/save_laws', methods=['POST'])
def save_laws():
    data = request.get_json(force=True)
    laws = data.get('laws', '')
    
    # Save with UTF-8 encoding to prevent character glitches
    laws_path = os.path.join('templates', 'system_laws.txt')
    with open(laws_path, 'w', encoding='utf-8') as f:
        f.write(laws)
        
    print("📜 System Laws updated by User.")
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # use_reloader=False prevents Flask from loading the model twice into VRAM
    app.run(debug=True, port=5002, use_reloader=False)
