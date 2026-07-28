# 🎨 SOM Color Palette Microservice API

![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MiniSom](https://img.shields.io/badge/MiniSom-SOM_Clustering-FF6F00?style=for-the-badge&logo=numpy&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deploy_Ready-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![Build Status](https://img.shields.io/badge/Tests-Passing-brightgreen?style=for-the-badge)

An asynchronous, production-ready REST API microservice that extracts topologically ordered color palettes and vector quantization metrics from images using **Self-Organizing Maps (SOM / Kohonen Neural Networks)**.

---

## 📌 Executive Overview

Unlike standard K-Means clustering which treats color clusters independently, **Self-Organizing Maps** maintain topological relationships between cluster centroids in a low-dimensional grid. This service maps high-dimensional image pixel vectors ($RGB \in \mathbb{R}^3$) onto a 2D neural lattice, producing smooth, visually cohesive color palettes alongside detailed convergence metrics.

### Key Features
* 🧠 **Unsupervised SOM Vector Quantization**: Continuous color space topology reduction via Gaussian neighborhood functions.
* ⚡ **Non-Blocking Asynchronous Processing**: CPU-intensive numpy/som matrix ops are offloaded to async threadpools via `asyncio.to_thread`.
* 🛡️ **Granular Permission & Access Control**: Built-in authorization middleware supporting **Client IP Whitelisting**, **Website Domain Origin Whitelisting**, and **API Key Headers (`X-API-Key`)**.
* 📊 **Structured Metrics & Logs**: Returns per-epoch Quantization Error ($QE$), Topographic Error ($TE$), Weight Delta ($\Delta W$), learning rate, neighborhood radius decay, and dominant color proportions.
* 🚀 **Render Blueprint & Docker Ready**: Pre-configured `render.yaml`, `Procfile`, and `Dockerfile` for instant zero-downtime deployment.

---

## 🧮 Mathematical & Algorithmic Foundation

### 1. Weight Vector Initialization
The SOM grid consists of $M \times N$ neurons. Each neuron $i$ maintains a weight vector $w_i \in \mathbb{R}^3$ initialized randomly within the pixel data range:
$$w_i(0) \sim \mathcal{U}(\min(X), \max(X))$$

### 2. Best Matching Unit (BMU) Selection
For each pixel vector $x \in \mathbb{R}^3$, the Best Matching Unit $c$ is determined via Euclidean distance minimization:
$$c = \arg\min_{i} \| x - w_i(t) \|$$

### 3. Exponential Parameter Decay
Learning rate $\alpha(t)$ and neighborhood radius $\sigma(t)$ decay exponentially over training epoch $t \in [0, T]$:
$$\alpha(t) = \alpha_0 \cdot \exp\left(-\frac{t}{T}\right), \quad \sigma(t) = \sigma_0 \cdot \exp\left(-\frac{t}{T}\right)$$

### 4. Topological Weight Update Rule
Weights are updated based on distance from the BMU in grid space $r_i, r_c$:
$$w_i(t+1) = w_i(t) + \alpha(t) \cdot h_{ci}(t) \cdot (x - w_i(t))$$
where the Gaussian neighborhood function $h_{ci}(t)$ is defined as:
$$h_{ci}(t) = \exp\left(-\frac{\| r_i - r_c \|^2}{2 \sigma(t)^2}\right)$$

### 5. Convergence Evaluation Metrics
* **Quantization Error ($QE$)**: Average distance from pixels to their assigned BMU weights:
  $$QE = \frac{1}{N} \sum_{k=1}^N \| x_k - w_{c(k)} \|$$
* **Topographic Error ($TE$)**: Proportion of sample vectors for which the 1st and 2nd BMUs are not adjacent on the grid lattice.

---

## 🛡️ Security & Access Control Configuration

The service features built-in access control enforced via environment variables. When no variables are set, the API operates in open mode.

| Environment Variable | Description | Example |
| :--- | :--- | :--- |
| `ALLOWED_IPS` | Comma-separated list of whitelisted client IP addresses. | `192.168.1.50, 10.0.4.12` |
| `ALLOWED_ORIGINS` | Comma-separated list of whitelisted domain origins. | `https://mywebsite.com, https://app.example.com` |
| `ALLOWED_API_KEYS` | Comma-separated list of valid API keys. | `sk_live_998877, sk_test_112233` |

### Authentication Headers
When API Key restriction is enabled, client requests must pass the `X-API-Key` header:
```bash
curl -X POST "https://your-som-service.onrender.com/api/v1/palette" \
  -H "X-API-Key: sk_live_998877" \
  -F "image=@photo.jpg"
```

---

## 📡 API Reference & Endpoints

### 1. `GET /health`
* **Summary**: Health check ping for Render / Uptime Robot.
* **Response**:
```json
{
  "status": "healthy",
  "service": "som-palette-api"
}
```

### 2. `POST /api/v1/palette`
* **Summary**: Extract structured color palette using SOM vector quantization.
* **Headers**: `X-API-Key` (Optional/Required based on config)
* **Form Parameters**:
  * `image` *(file, required)*: JPEG/PNG/WEBP image file.
  * `grid_x` *(int, default: 3)*: SOM grid width (1-10).
  * `grid_y` *(int, default: 3)*: SOM grid height (1-10).
  * `max_epochs` *(int, default: 100)*: Max training epochs (5-500).
  * `tolerance` *(float, default: 0.02)*: ΔW convergence threshold.
  * `patience_limit` *(int, default: 3)*: Epoch patience limit for early stopping.
  * `initial_lr` *(float, default: 0.1)*: Initial learning rate $\alpha_0$.
  * `initial_sigma` *(float, default: 1.5)*: Initial neighborhood radius $\sigma_0$.

* **Example Response (`200 OK`)**:
```json
{
  "palette": [
    {
      "index": 0,
      "rgb": [212, 145, 98],
      "hex": "#d49162",
      "percentage": 28.45,
      "pixel_count": 11380
    },
    {
      "index": 1,
      "rgb": [42, 65, 88],
      "hex": "#2a4158",
      "percentage": 22.10,
      "pixel_count": 8840
    }
  ],
  "metrics": [
    {
      "epoch": 1,
      "weight_delta": 0.0421589,
      "quantization_error": 0.081245,
      "topographic_error": 0.015,
      "learning_rate": 0.099,
      "sigma": 1.485
    }
  ],
  "summary": {
    "image_resolution": "800x600",
    "grid_dimensions": "3x3",
    "total_colors": 9,
    "total_epochs_run": 24,
    "final_quantization_error": 0.041203,
    "final_topographic_error": 0.0,
    "stop_reason": "Converged: ΔW (0.01892) reached tolerance threshold (< 0.02)."
  }
}