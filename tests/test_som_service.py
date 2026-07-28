import io
import numpy as np
from PIL import Image
from som_service import generate_palette_with_minisom, rgb_to_hex

def create_synthetic_image():
    """Generates a small 50x50 synthetic RGB image byte buffer."""
    img_data = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    img = Image.fromarray(img_data)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def test_rgb_to_hex():
    assert rgb_to_hex([255, 0, 0]) == "#ff0000"
    assert rgb_to_hex([0, 255, 0]) == "#00ff00"
    assert rgb_to_hex([0, 0, 255]) == "#0000ff"
    assert rgb_to_hex([255, 255, 255]) == "#ffffff"

def test_som_palette_generation():
    img_buf = create_synthetic_image()
    res = generate_palette_with_minisom(
        image_file=img_buf,
        grid_x=3,
        grid_y=2,
        max_epochs=10,
        tolerance=0.05,
        patience_limit=2
    )

    assert "palette" in res
    assert "metrics" in res
    assert "logs" in res
    assert "summary" in res

    # 3x2 grid should output 6 colors
    assert len(res["palette"]) == 6
    assert res["summary"]["total_colors"] == 6

    # Verify color structure
    first_color = res["palette"][0]
    assert "rgb" in first_color
    assert "hex" in first_color
    assert "percentage" in first_color
    assert len(first_color["rgb"]) == 3
    assert first_color["hex"].startswith("#")

    # Verify metrics structure
    assert len(res["metrics"]) > 0
    metric = res["metrics"][0]
    assert "epoch" in metric
    assert "weight_delta" in metric
    assert "quantization_error" in metric
    assert "topographic_error" in metric
