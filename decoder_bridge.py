import zlib
import os

class AIDecoderBridge:
    def __init__(self, kb_dir='Knowledge_Base'):
        self.kb_dir = kb_dir

    def get_binary_content(self, category, filename):
        """Finds the .dat file in the specified category folder"""
        path = os.path.join(self.kb_dir, category, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return f.read()
        print(f"❌ File not found: {path}")
        return None

    def decode_for_ai(self, binary_data):
        """Turns the binary into text so the AI can understand the 'Truth'"""
        try:
            # This is your core decoder logic
            decompressed = zlib.decompress(binary_data)
            return decompressed.decode('utf-8', errors='replace')
        except Exception as e:
            return f"DECODE_ERROR: {e}"

    def prepare_training_data(self, category, filename):
        """The AI calls this to get both the 'Binary' and the 'Meaning'"""
        binary_content = self.get_binary_content(category, filename)
        if not binary_content: return None

        decoded_text = self.decode_for_ai(binary_content)

        return {
            "binary_raw": list(binary_content), # Pure bits
            "decoded_text": decoded_text,       # Human-readable code
            "category": category,
            "filename": filename
        }

# --- QUICK TEST ---
if __name__ == "__main__":
    bridge = AIDecoderBridge()
    # Replace 'Python' and 'test.dat' with a real folder/file you have
    test_data = bridge.prepare_training_data('Python', 'python_data_1.dat')
    if test_data:
        print(f"✅ Bridge Ready for {test_data['filename']}")
        print(f"📖 Decoded Logic: {test_data['decoded_text'][:50]}...")
