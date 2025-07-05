"""
AI Tax Advisor Demo - Standalone Implementation
FastAPI application serving only the AI demo interface
"""

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

# Import AI routes
from api.ai_routes import router as ai_router

# Create FastAPI app
app = FastAPI(
    title="AI Tax Advisor Demo",
    description="Standalone AI Tax Advisor Demo using Gemini Flash 2.0 Pro",
    version="1.0.0"
)

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="ai-demo-secret-key")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Include AI routes
app.include_router(ai_router)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to AI demo"""
    return templates.TemplateResponse("ai_demo.html", {"request": request})

@app.get("/ai-demo", response_class=HTMLResponse)
async def ai_demo(request: Request):
    """AI integration demo page"""
    return templates.TemplateResponse("ai_demo.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Tax Advisor Demo"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 