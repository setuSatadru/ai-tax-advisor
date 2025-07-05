"""
Settings configuration for AI Tax Advisor Demo
Environment-based configuration management for standalone AI demo
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings for AI Tax Advisor Demo"""
    
    # Application settings
    APP_NAME: str = "AI Tax Advisor Demo"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Security settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "ai-demo-secret-key")
    
    # AI/LLM settings - Gemini Flash 2.0 Pro
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    GEMINI_MAX_TOKENS: int = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
    
    # API Rate Limiting settings - Updated for free tier
    GEMINI_REQUESTS_PER_MINUTE: int = int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "15"))  # Free tier limit
    GEMINI_REQUESTS_PER_HOUR: int = int(os.getenv("GEMINI_REQUESTS_PER_HOUR", "900"))     # Conservative daily limit
    GEMINI_MAX_RETRIES: int = int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    GEMINI_RETRY_DELAY: int = int(os.getenv("GEMINI_RETRY_DELAY", "1"))
    
    # AI Safety settings
    GEMINI_SAFETY_THRESHOLD: str = os.getenv("GEMINI_SAFETY_THRESHOLD", "BLOCK_MEDIUM_AND_ABOVE")
    GEMINI_RESPONSE_VALIDATION: bool = os.getenv("GEMINI_RESPONSE_VALIDATION", "true").lower() == "true"
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "ai_demo.log")
    
    def __init__(self):
        """Initialize settings and create necessary directories"""
        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        # Validate Gemini API configuration
        if not self.GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not set. AI features will be disabled.")
            return False
        
        # Validate rate limiting settings
        if self.GEMINI_REQUESTS_PER_MINUTE <= 0 or self.GEMINI_REQUESTS_PER_HOUR <= 0:
            raise ValueError("Invalid rate limiting configuration")
        
        return True

# Global settings instance
settings = Settings()

# Validate configuration on import
try:
    settings.validate_config()
except Exception as e:
    print(f"Configuration validation error: {e}")
    raise 