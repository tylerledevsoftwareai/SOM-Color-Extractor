import io
import os
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from main import app
from auth import access_control

client = TestClient(app)

def create_synthetic_image_bytes():
    img_data = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
    img = Image.fromarray(img_data)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf.getvalue()

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Self-Organizing Map (SOM) Palette Service"
    assert data["status"] == "online"

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "som-palette-api"}

def test_extract_palette_success():
    img_bytes = create_synthetic_image_bytes()
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
    data = {"grid_x": 2, "grid_y": 2, "max_epochs": 5}
    
    response = client.post("/api/v1/palette", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert "palette" in res_json
    assert len(res_json["palette"]) == 4

def test_invalid_file_format():
    files = {"image": ("text.txt", b"not an image", "text/plain")}
    response = client.post("/api/v1/palette", files=files)
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

def test_access_control_api_key_restriction(monkeypatch):
    monkeypatch.setenv("ALLOWED_API_KEYS", "secret-key-123")
    access_control._reload_permissions()

    img_bytes = create_synthetic_image_bytes()
    files = {"image": ("test.jpg", img_bytes, "image/jpeg")}

    # 1. Without API key -> 403 Forbidden
    response = client.post("/api/v1/palette", files=files)
    assert response.status_code == 403

    # 2. With invalid API key -> 403 Forbidden
    headers = {"X-API-Key": "wrong-key"}
    response = client.post("/api/v1/palette", files=files, headers=headers)
    assert response.status_code == 403

    # 3. With valid API key -> 200 OK
    headers = {"X-API-Key": "secret-key-123"}
    data = {"grid_x": 2, "grid_y": 2, "max_epochs": 5}
    response = client.post("/api/v1/palette", files=files, data=data, headers=headers)
    assert response.status_code == 200

    # Reset permissions
    monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
    access_control._reload_permissions()
