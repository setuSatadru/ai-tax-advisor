"""
Gemini Flash 2.0 Pro Service for Tax Advisor Application
Handles all interactions with Google's Gemini AI API
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from config.settings import settings


# Configure logging
logger = logging.getLogger(__name__)


class ResponseType(Enum):
    """Types of AI responses"""
    TAX_SUGGESTIONS = "tax_suggestions"
    REGIME_COMPARISON = "regime_comparison"
    INVESTMENT_ADVICE = "investment_advice"
    DEDUCTION_OPTIMIZATION = "deduction_optimization"
    CHAT_RESPONSE = "chat_response"


@dataclass
class GeminiRequest:
    """Request structure for Gemini API calls"""
    prompt: str
    response_type: ResponseType
    context: Dict[str, Any]
    session_id: str
    user_id: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class GeminiResponse:
    """Response structure from Gemini API"""
    content: str
    response_type: ResponseType
    confidence_score: float
    processing_time: float
    session_id: str
    timestamp: datetime
    tokens_used: int
    is_valid: bool
    error_message: Optional[str] = None


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, requests_per_minute: int, requests_per_hour: int):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests = []
        self.hour_requests = []
    
    def can_make_request(self) -> bool:
        """Check if a request can be made within rate limits"""
        now = datetime.now()
        
        # Clean old requests
        self.minute_requests = [req_time for req_time in self.minute_requests 
                               if now - req_time < timedelta(minutes=1)]
        self.hour_requests = [req_time for req_time in self.hour_requests 
                             if now - req_time < timedelta(hours=1)]
        
        # Check limits
        if len(self.minute_requests) >= self.requests_per_minute:
            return False
        if len(self.hour_requests) >= self.requests_per_hour:
            return False
        
        return True
    
    def record_request(self):
        """Record a new request"""
        now = datetime.now()
        self.minute_requests.append(now)
        self.hour_requests.append(now)
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request"""
        if not self.minute_requests and not self.hour_requests:
            return 0
        
        now = datetime.now()
        
        # Check minute limit
        if len(self.minute_requests) >= self.requests_per_minute:
            oldest_minute = min(self.minute_requests)
            wait_time_minute = 60 - (now - oldest_minute).total_seconds()
            if wait_time_minute > 0:
                return wait_time_minute
        
        # Check hour limit
        if len(self.hour_requests) >= self.requests_per_hour:
            oldest_hour = min(self.hour_requests)
            wait_time_hour = 3600 - (now - oldest_hour).total_seconds()
            if wait_time_hour > 0:
                return wait_time_hour
        
        return 0


class GeminiService:
    """Service for interacting with Gemini Flash 2.0 Pro API"""
    
    def __init__(self):
        """Initialize Gemini service with configuration"""
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL
        self.max_tokens = settings.GEMINI_MAX_TOKENS
        self.temperature = settings.GEMINI_TEMPERATURE
        self.max_retries = settings.GEMINI_MAX_RETRIES
        self.retry_delay = settings.GEMINI_RETRY_DELAY
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            settings.GEMINI_REQUESTS_PER_MINUTE,
            settings.GEMINI_REQUESTS_PER_HOUR
        )
        
        # Initialize model
        self.model = None
        self.is_initialized = False
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Initialize if API key is available
        if self.api_key:
            self.initialize()
    
    def initialize(self) -> bool:
        """Initialize Gemini API connection"""
        try:
            if not self.api_key:
                logger.error("Gemini API key not configured")
                return False
            
            # Configure API
            genai.configure(api_key=self.api_key)
            
            # Initialize model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings
            )
            
            self.is_initialized = True
            logger.info(f"Gemini service initialized with model: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini service: {e}")
            self.is_initialized = False
            return False
    
    def is_available(self) -> bool:
        """Check if Gemini service is available"""
        return self.is_initialized and self.model is not None
    
    async def generate_response(self, request: GeminiRequest) -> GeminiResponse:
        """Generate response from Gemini API with error handling and retries"""
        start_time = time.time()
        
        if not self.is_available():
            return GeminiResponse(
                content="AI service is not available",
                response_type=request.response_type,
                confidence_score=0.0,
                processing_time=0.0,
                session_id=request.session_id,
                timestamp=datetime.now(),
                tokens_used=0,
                is_valid=False,
                error_message="Gemini service not initialized"
            )
        
        # Check rate limits
        if not self.rate_limiter.can_make_request():
            wait_time = self.rate_limiter.get_wait_time()
            return GeminiResponse(
                content="Rate limit exceeded. Please try again later.",
                response_type=request.response_type,
                confidence_score=0.0,
                processing_time=0.0,
                session_id=request.session_id,
                timestamp=datetime.now(),
                tokens_used=0,
                is_valid=False,
                error_message=f"Rate limit exceeded. Wait {wait_time:.0f} seconds"
            )
        
        # Attempt to generate response with retries
        for attempt in range(self.max_retries + 1):
            try:
                # Record request for rate limiting
                self.rate_limiter.record_request()
                
                # Generate content
                response = await self._make_api_call(request)
                
                processing_time = time.time() - start_time
                
                # Validate response
                is_valid, error_message = self._validate_response(response, request.response_type)
                
                # Calculate confidence score (simplified)
                confidence_score = self._calculate_confidence_score(response, request.response_type)
                
                return GeminiResponse(
                    content=response.text if response.text else "No response generated",
                    response_type=request.response_type,
                    confidence_score=confidence_score,
                    processing_time=processing_time,
                    session_id=request.session_id,
                    timestamp=datetime.now(),
                    tokens_used=self._estimate_tokens(response.text if response.text else ""),
                    is_valid=is_valid,
                    error_message=error_message
                )
                
            except Exception as e:
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    processing_time = time.time() - start_time
                    return GeminiResponse(
                        content="Failed to generate AI response",
                        response_type=request.response_type,
                        confidence_score=0.0,
                        processing_time=processing_time,
                        session_id=request.session_id,
                        timestamp=datetime.now(),
                        tokens_used=0,
                        is_valid=False,
                        error_message=str(e)
                    )
    
    async def _make_api_call(self, request: GeminiRequest):
        """Make the actual API call to Gemini"""
        generation_config = genai.types.GenerationConfig(
            temperature=request.temperature or self.temperature,
            max_output_tokens=request.max_tokens or self.max_tokens,
        )
        
        response = await self.model.generate_content_async(
            request.prompt,
            generation_config=generation_config,
            safety_settings=self.safety_settings
        )
        
        return response
    
    def _validate_response(self, response, response_type: ResponseType) -> Tuple[bool, Optional[str]]:
        """Validate AI response based on type and content"""
        if not response or not response.text:
            return False, "Empty response from AI"
        
        # Check for safety blocks
        if response.prompt_feedback:
            if response.prompt_feedback.block_reason:
                return False, f"Content blocked: {response.prompt_feedback.block_reason}"
        
        # Basic content validation
        content = response.text.strip()
        if len(content) < 10:
            return False, "Response too short"
        
        # Type-specific validation
        if response_type == ResponseType.TAX_SUGGESTIONS:
            if not any(keyword in content.lower() for keyword in 
                      ['tax', 'deduction', 'saving', 'investment', 'section']):
                return False, "Response doesn't contain tax-related content"
        
        elif response_type == ResponseType.REGIME_COMPARISON:
            if not any(keyword in content.lower() for keyword in 
                      ['old regime', 'new regime', 'comparison', 'better']):
                return False, "Response doesn't contain regime comparison"
        
        return True, None
    
    def _calculate_confidence_score(self, response, response_type: ResponseType) -> float:
        """Calculate confidence score for the response"""
        if not response or not response.text:
            return 0.0
        
        # Basic confidence calculation
        base_score = 0.7
        
        # Adjust based on response length
        length_factor = min(len(response.text) / 500, 1.0) * 0.2
        
        # Adjust based on content quality (simplified)
        content_lower = response.text.lower()
        quality_keywords = ['specific', 'recommend', 'suggest', 'consider', 'analysis']
        quality_factor = sum(1 for keyword in quality_keywords if keyword in content_lower) * 0.02
        
        confidence_score = min(base_score + length_factor + quality_factor, 1.0)
        return round(confidence_score, 2)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for the response"""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            "is_available": self.is_available(),
            "model_name": self.model_name,
            "rate_limits": {
                "requests_per_minute": self.rate_limiter.requests_per_minute,
                "requests_per_hour": self.rate_limiter.requests_per_hour,
                "current_minute_requests": len(self.rate_limiter.minute_requests),
                "current_hour_requests": len(self.rate_limiter.hour_requests)
            },
            "configuration": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay
            }
        }
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test Gemini API connection"""
        if not self.is_available():
            return {
                "success": False,
                "error": "Service not initialized",
                "details": "Gemini API key not configured or service failed to initialize"
            }
        
        try:
            test_request = GeminiRequest(
                prompt="Hello, this is a test. Please respond with 'Test successful'.",
                response_type=ResponseType.CHAT_RESPONSE,
                context={},
                session_id="test_session"
            )
            
            response = await self.generate_response(test_request)
            
            return {
                "success": response.is_valid,
                "response_time": response.processing_time,
                "tokens_used": response.tokens_used,
                "error": response.error_message
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "details": "Failed to test Gemini API connection"
            }


# Global service instance
gemini_service = GeminiService() 