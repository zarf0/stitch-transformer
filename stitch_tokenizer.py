# stitch_tokenizer.py
import numpy as np
import pyembroidery
import os
import json

class StitchTokenizer:
    def __init__(self, max_seq_len=20000):
        self.max_seq_len = max_seq_len
        self.start_token = [0, 0, 0, 0, 0]      # Start of design
        self.end_token = [0, 0, 0, 0, 1]        # End of design
        self.pad_token = [0, 0, 0, 0, 0]
        
    def parse_file(self, filepath):
        """Parse embroidery file and return stitch tokens."""
        try:
            pattern = pyembroidery.read(filepath)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
            
        if pattern is None:
            return None
            
        STITCH = int(getattr(pyembroidery, "STITCH", 0))
        JUMP = int(getattr(pyembroidery, "JUMP", 1))
        TRIM = int(getattr(pyembroidery, "TRIM", 2))
        COLOR_CHANGE = int(getattr(pyembroidery, "COLOR_CHANGE", 4))
        END = int(getattr(pyembroidery, "END", 5))
        COMMAND_MASK = int(getattr(pyembroidery, "COMMAND_MASK", 0xFF))
        
        tokens = [self.start_token]
        
        prev_x, prev_y = 0, 0
        color_idx = 0
        threads = getattr(pattern, "threadlist", []) or []
        
        for stitch in getattr(pattern, "stitches", []):
            if len(stitch) < 3:
                continue
                
            x, y, raw_cmd = stitch
            try:
                cmd = int(raw_cmd) & COMMAND_MASK
            except:
                cmd = int(raw_cmd)
            
            # Convert to mm (divide by 10)
            x_mm = float(x) / 10.0
            y_mm = float(y) / 10.0
            
            # Calculate delta
            dx = x_mm - prev_x
            dy = y_mm - prev_y
            
            if cmd == STITCH:
                token = [dx, dy, 1, 0, 0]  # Is_Stitch=1
                prev_x, prev_y = x_mm, y_mm
                tokens.append(token)
                
            elif cmd == JUMP:
                token = [dx, dy, 0, 0, 0]  # Is_Stitch=0
                prev_x, prev_y = x_mm, y_mm
                tokens.append(token)
                
            elif cmd == TRIM:
                token = [dx, dy, 0, 1, 0]  # Is_Trim=1
                prev_x, prev_y = x_mm, y_mm
                tokens.append(token)
                
            elif cmd == COLOR_CHANGE:
                token = [0, 0, 0, 0, 1]  # Is_Color_Change=1
                color_idx += 1
                tokens.append(token)
                
            elif cmd == END:
                break
                
        tokens.append(self.end_token)
        
        # Truncate if too long
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            tokens[-1] = self.end_token
            
        return tokens
    
    def tokenize_file(self, filepath):
        return self.parse_file(filepath)
    
    def save_tokens(self, tokens, filepath):
        with open(filepath, 'w') as f:
            json.dump(tokens, f)
    
    def load_tokens(self, filepath):
        with open(filepath, 'r') as f:
            return json.load(f)