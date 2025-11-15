# --- FIX in Main Loop ---
from utah_semantic_indexer_fixed import compress 

# ... (short_side_scale and center_crop definitions) ...

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Apply transforms: scale + crop."""
    scaled = short_side_scale(frame, side_size)
    cropped = center_crop(scaled, crop_size)
    return cropped.astype(np.float32) # Ensure float type for stack

# --- Preprocess + Compress ---
def preprocess_and_compress(frames: list) -> str:
    """Preprocess, stack, and compress video frames."""
    
    # 1. Preprocess frames
    processed = [preprocess_frame(f) for f in frames]
    
    # 2. Stack into 4D array (T, H, W, C)
    stack = np.stack(processed)
    
    # 3. Compress using the fixed semantic indexer
    vector = compress(stack) 
    
    # 4. Base64 encode for API transmission
    return base64.b64encode(vector).decode('utf-8') 