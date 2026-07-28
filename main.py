import io
import asyncio
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from som_service import generate_palette_with_minisom
from auth import verify_permissions, access_control

app = FastAPI(
    title="SOM Palette Extraction Microservice",
    description="""
## Self-Organizing Map (SOM) Image Palette API Service

Extract high-fidelity color palettes and topological vector quantization metrics from images using **Self-Organizing Maps (Kohonen Networks)**.

### Features
* **Unsupervised Clustering**: Uses 2D Self-Organizing Map topology to sample continuous RGB color spaces.
* **Full-Resolution Unique Pixel Deduplication**: Trains on all distinct pixel values without artificial downsampling or sample limits.
* **Stateful Training & Convergence**: Exponential learning rate and neighborhood radius decay with quantization & topographic error tracking.
* **Access Control & Permissions**: IP Address, Origin Domain, and API Key authorization rules.
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
def root(request: Request):
    """
    Service information and interactive API documentation links.
    """
    client_ip = access_control.extract_client_ip(request)
    return {
        "service": "Self-Organizing Map (SOM) Palette Service",
        "status": "online",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "your_client_ip": client_ip
    }

@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint for Render zero-downtime ping monitoring.
    """
    return {"status": "healthy", "service": "som-palette-api"}

@app.post(
    "/api/v1/palette",
    tags=["SOM Palette Extraction"],
    dependencies=[Depends(verify_permissions)],
    summary="Extract Color Palette using SOM Vector Quantization"
)
async def extract_palette(
    image: UploadFile = File(..., description="Target image file (JPEG, PNG, WEBP, BMP)"),
    grid_x: int = Form(3, ge=1, le=10, description="SOM grid width (X axis size)"),
    grid_y: int = Form(3, ge=1, le=10, description="SOM grid height (Y axis size)"),
    max_epochs: int = Form(100, ge=5, le=500, description="Maximum training epochs"),
    tolerance: float = Form(0.02, ge=0.0001, le=0.5, description="Convergence ΔW tolerance threshold"),
    patience_limit: int = Form(3, ge=1, le=20, description="Epoch patience limit for quantization error stabilization"),
    initial_lr: float = Form(0.1, ge=0.01, le=1.0, description="Initial learning rate (α₀)"),
    initial_sigma: float = Form(1.5, ge=0.1, le=5.0, description="Initial neighborhood radius (σ₀)"),
    min_color_distance: float = Form(0.0, ge=0.0, le=150.0, description="Optional minimum Euclidean distance threshold (in RGB space 0-255) to merge similar colors (0.0 to disable)")
):
    """
    **Extracts a color palette from an uploaded image using MiniSom.**
    
    Requires access permissions (valid Client IP, whitelisted Origin header, or `X-API-Key` header if security rules are enabled).
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Uploaded file must be an image (e.g. image/jpeg, image/png)."
        )

    try:
        image_bytes = await image.read()
        image_file = io.BytesIO(image_bytes)

        # Offload CPU-bound SOM vector quantization to async threadpool to keep event loop responsive
        result = await asyncio.to_thread(
            generate_palette_with_minisom,
            image_file=image_file,
            grid_x=grid_x,
            grid_y=grid_y,
            max_epochs=max_epochs,
            tolerance=tolerance,
            patience_limit=patience_limit,
            initial_lr=initial_lr,
            initial_sigma=initial_sigma,
            min_color_distance=min_color_distance
        )

        return JSONResponse(status_code=status.HTTP_200_OK, content=result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the image: {str(e)}"
        )
