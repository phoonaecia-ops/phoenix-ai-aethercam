# utah_semantic_indexer_fixed.py
# Purpose: Compress Video Binary Data → 8.06KB Semantic Soul Vector
# NOTE: Requires PyTorch, torchvision, and pytorchvideo to be installed.

import torch
import hashlib
import numpy as np
from pytorchvideo.models.hub import x3d_m  # Lightweight 3D CNN for feature extraction
import torch.nn as nn
import os
import io

# --- CONFIG ---
# Target Embedding Size: We aim for a manageable float vector (e.g., 2048 dims)
# which is then compressed to the 8060 bytes (8.06 KB) limit.
EMBEDDING_SIZE = 2048
TARGET_BYTE_SIZE = 8060 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the X3D model, stripped down to its feature extractor
try:
    # Load X3D-M pretrained on Kinetics, but use it to output features
    model = x3d_m(pretrained=True).eval().to(device)
    # Strip the final classification layer to get the feature vector (e.g., 2048 dims)
    if hasattr(model, 'blocks'):
        # This is a common way to access the feature head in PyTorchVideo models
        model.blocks[-1] = nn.Identity() 
    else:
        # Fallback/alternative way to remove the final classification layer
        model.head = nn.Identity()
        
except Exception as e:
    print(f"Error loading PyTorchVideo model: {e}")
    print("WARNING: Using a mock feature extractor. Install pytorchvideo for real features.")
    class MockModel(nn.Module):
        def forward(self, x): return torch.randn(x.size(0), EMBEDDING_SIZE)
    model = MockModel().to(device)


# --- CORE COMPRESSION (Video-Specific) ---
def compress(data_stack: np.ndarray) -> bytes:
    """
    Compress a 4D NumPy array (T, H, W, C) representing video frames 
    into an 8.06KB semantic soul vector.
    
    Args:
        data_stack (np.ndarray): Video clip stack, expected (8, 224, 224, 3) float.

    Steps:
    1. Convert NumPy stack (T, H, W, C) → PyTorch tensor (N, C, T, H, W).
    2. Extract Features using 3D CNN (X3D) → (N, EMBEDDING_SIZE).
    3. Quantize the float vector (Lossy Compression) and convert to bytes.
    4. Truncate/Pad to precisely 8060 bytes (8.06KB).
    """
    if data_stack.ndim != 4:
        raise ValueError("Input must be a 4D NumPy array (T, H, W, C) of video frames.")

    # 1. Convert to PyTorch Tensor format: (T, H, W, C) -> (N=1, C, T, H, W)
    # Normalize to [0, 1] range
    video_tensor = torch.from_numpy(data_stack).permute(3, 0, 1, 2).float() / 255.0
    input_batch = video_tensor.unsqueeze(0).to(device)
    
    # 2. Extract Features
    with torch.no_grad():
        # X3D outputs the feature vector
        features = model(input_batch).squeeze().cpu().numpy() # (EMBEDDING_SIZE,)
    
    # --- 3. Quantization and Encoding (Lossy Compression) ---
    # Convert the float features into a compact, fixed-size byte string.
    # We'll use float16 precision and then pack/pad to the target size.
    
    # Convert 32-bit floats to 16-bit floats for 2x compression
    compressed_vector = features.astype(np.float16)
    vector_bytes = compressed_vector.tobytes()
    
    # --- 4. Truncate/Pad to 8.06KB ---
    if len(vector_bytes) > TARGET_BYTE_SIZE:
        # Truncate the high-dimensional feature vector if it's too large
        soul_vector = vector_bytes[:TARGET_BYTE_SIZE]
    else:
        # Pad with zeros if it's smaller (ensures fixed payload size)
        soul_vector = vector_bytes + b'\x00' * (TARGET_BYTE_SIZE - len(vector_bytes))
        
    return soul_vector

# --- USAGE EXAMPLE ---
if __name__ == "__main__":
    # Simulate a stack of 8 pre-processed (224x224x3) frames
    dummy_stack = np.random.rand(8, 224, 224, 3).astype(np.float32)
    soul = compress(dummy_stack)
    
    # The output will be exactly 8060 bytes
    print(f"8 Frames → {len(soul) / 1024:.2f} KB (Target: 8.06 KB)")
    print(f"SOUL HASH (first 16 bytes): {soul[:16].hex()}")