import numpy as np
from PIL import Image
from minisom import MiniSom
import io
from typing import Dict, Any, Tuple, List, Union

def rgb_to_hex(rgb: Union[List[int], Tuple[int, int, int]]) -> str:
    """Converts RGB integers [R, G, B] to Hexadecimal color string format '#RRGGBB'."""
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"

def generate_palette_with_minisom(
    image_file: Any,
    grid_x: int = 3,
    grid_y: int = 3,
    max_epochs: int = 100,
    tolerance: float = 0.02,
    patience_limit: int = 3,
    qe_improvement_threshold: float = 0.002,
    initial_lr: float = 0.1,
    initial_sigma: float = 1.5,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Extracts a color palette from an image using a Self-Organizing Map (SOM).
    
    :param image_file: A file-like object or byte stream containing the image data.
    :param grid_x: SOM grid width (number of palette colors along X axis).
    :param grid_y: SOM grid height (number of palette colors along Y axis).
    :param max_epochs: Maximum number of training epochs.
    :param tolerance: Threshold for weight changes (ΔW) to determine convergence.
    :param patience_limit: Number of epochs to wait for quantization error stability before early stopping.
    :param qe_improvement_threshold: Minimum required relative improvement ratio in quantization error.
    :param initial_lr: Initial learning rate (α₀).
    :param initial_sigma: Initial neighborhood radius (σ₀).
    :param random_seed: Random seed for deterministic initialization.
    :return: A dictionary containing extracted palette swatches, convergence metrics per epoch, logs, and summary.
    """
    # 1. Image Preprocessing
    img = Image.open(image_file).convert('RGB')
    width, height = img.size
    
    # Downsample large images for fast & memory-efficient SOM training while maintaining color fidelity
    max_dim = 400
    if max(width, height) > max_dim:
        scale = max_dim / float(max(width, height))
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    img_data = np.array(img) / 255.0
    pixels = img_data.reshape(-1, 3)

    # 2. SOM Initialization
    som = MiniSom(
        x=grid_x, 
        y=grid_y, 
        input_len=3, 
        sigma=initial_sigma, 
        learning_rate=initial_lr,
        neighborhood_function='gaussian', 
        random_seed=random_seed
    )
    som.random_weights_init(pixels)

    # 3. Stateful Training Loop with Exponential Decay
    logs: List[str] = [f"Starting incremental training on {len(pixels)} pixel vectors (Grid: {grid_x}x{grid_y}, max {max_epochs} epochs)..."]
    epoch_metrics: List[Dict[str, Any]] = []

    prev_qe = None
    patience_counter = 0
    stop_reason = f"Completed maximum {max_epochs} epochs."

    for epoch in range(max_epochs):
        # Capture w_i(t-1)
        old_weights = som.get_weights().reshape(-1, 3).copy()

        # Apply exponential decay formulas
        t = epoch
        T = max_epochs
        som.learning_rate = initial_lr * np.exp(-t / T)
        som.sigma = initial_sigma * np.exp(-t / T)

        # Train for one full pass over the dataset
        som.train_random(pixels, len(pixels), verbose=False)

        # Capture w_i(t)
        current_weights = som.get_weights().reshape(-1, 3)

        # Compute convergence metrics
        weight_delta = float(np.mean(np.linalg.norm(current_weights - old_weights, axis=1)))
        q_error = float(som.quantization_error(pixels))
        t_error = float(som.topographic_error(pixels))

        metric_record = {
            "epoch": epoch + 1,
            "weight_delta": round(weight_delta, 8),
            "quantization_error": round(q_error, 6),
            "topographic_error": round(t_error, 4),
            "learning_rate": round(float(som.learning_rate), 4),
            "sigma": round(float(som.sigma), 4)
        }
        epoch_metrics.append(metric_record)

        log_str = (f"Epoch {epoch+1:02d} | ΔW: {weight_delta:.8f} | QE: {q_error:.6f} | "
                   f"TE: {t_error:.4f} | LR: {som.learning_rate:.4f} | Sig: {som.sigma:.4f}")
        logs.append(log_str)

        # --- Early Stopping Criteria ---
        if weight_delta < tolerance:
            stop_reason = f"Converged: ΔW ({weight_delta:.5f}) reached tolerance threshold (< {tolerance})."
            logs.append(f"\nStopping: {stop_reason}")
            break

        if prev_qe is not None:
            improvement = (prev_qe - q_error) / prev_qe if prev_qe > 0 else 0
            if improvement < qe_improvement_threshold:
                patience_counter += 1
            else:
                patience_counter = 0

        if patience_counter >= patience_limit:
            stop_reason = f"Patience limit reached: QE improvement < {qe_improvement_threshold * 100}% for {patience_limit} consecutive epochs."
            logs.append(f"\nStopping: {stop_reason}")
            break

        prev_qe = q_error

    # 4. Color Swatch Extraction & Winning Node Counts
    final_weights = som.get_weights().reshape(-1, 3)
    rgb_weights = np.clip(final_weights * 255, 0, 255).astype(int)

    # Compute winning neuron frequency to determine dominant color percentages
    winner_coordinates = [som.winner(p) for p in pixels]
    winner_indices = [x * grid_y + y for (x, y) in winner_coordinates]
    counts = np.bincount(winner_indices, minlength=grid_x * grid_y)
    total_pixels = len(pixels)

    palette = []
    for idx, rgb in enumerate(rgb_weights):
        rgb_list = rgb.tolist()
        hex_code = rgb_to_hex(rgb_list)
        pixel_count = int(counts[idx])
        percentage = round((pixel_count / total_pixels) * 100, 2)
        
        palette.append({
            "index": idx,
            "rgb": rgb_list,
            "hex": hex_code,
            "percentage": percentage,
            "pixel_count": pixel_count
        })

    # Sort palette by dominance percentage descending
    palette.sort(key=lambda x: x["percentage"], reverse=True)

    summary = {
        "image_resolution": f"{width}x{height}",
        "grid_dimensions": f"{grid_x}x{grid_y}",
        "total_colors": len(palette),
        "total_epochs_run": len(epoch_metrics),
        "final_quantization_error": epoch_metrics[-1]["quantization_error"] if epoch_metrics else 0.0,
        "final_topographic_error": epoch_metrics[-1]["topographic_error"] if epoch_metrics else 0.0,
        "stop_reason": stop_reason
    }

    return {
        "palette": palette,
        "metrics": epoch_metrics,
        "logs": logs,
        "summary": summary
    }
