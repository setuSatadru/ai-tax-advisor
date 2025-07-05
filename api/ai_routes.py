"""
AI Routes for Tax Advisor Application
API endpoints for Gemini Flash 2.0 Pro integration
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

# Import our services
from services import (
    GeminiService, GeminiRequest, GeminiResponse, ResponseType,
    TaxPromptTemplates, PromptType,
    ResponseParser, ParsedResponse, response_parser
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/ai", tags=["AI Tax Advisor"])

# Initialize services
gemini_service = GeminiService()
prompt_templates = TaxPromptTemplates()
parser = ResponseParser()


# Pydantic models for API requests/responses
class TaxSuggestionsRequest(BaseModel):
    """Request model for tax suggestions"""
    basic_salary: float = Field(..., description="Basic salary amount")
    hra: float = Field(..., description="HRA amount")
    other_allowances: float = Field(..., description="Other allowances")
    gross_salary: float = Field(..., description="Gross salary")
    current_deductions: float = Field(..., description="Current deductions")
    net_salary: float = Field(..., description="Net salary")
    age: int = Field(..., description="Age of the taxpayer")
    marital_status: str = Field(..., description="Marital status")
    dependents: int = Field(..., description="Number of dependents")
    city: str = Field(..., description="City of residence")
    rent_paid: float = Field(..., description="Monthly rent paid")
    current_investments: str = Field(..., description="Current investments")
    financial_goals: str = Field(..., description="Financial goals")
    session_id: str = Field(..., description="Session ID")


class RegimeComparisonRequest(BaseModel):
    """Request model for regime comparison"""
    gross_salary: float = Field(..., description="Gross salary")
    basic_salary: float = Field(..., description="Basic salary")
    hra: float = Field(..., description="HRA amount")
    other_allowances: float = Field(..., description="Other allowances")
    section_80c: float = Field(..., description="Section 80C investments")
    section_80d: float = Field(..., description="Section 80D investments")
    home_loan_interest: float = Field(..., description="Home loan interest")
    other_deductions: float = Field(..., description="Other deductions")
    age: int = Field(..., description="Age")
    city: str = Field(..., description="City")
    rent_paid: float = Field(..., description="Monthly rent")
    session_id: str = Field(..., description="Session ID")


class ChatRequest(BaseModel):
    """Request model for chat"""
    user_query: str = Field(..., description="User query")
    session_id: str = Field(..., description="Session ID")
    previous_context: Optional[str] = Field(None, description="Previous context")
    user_profile: Optional[str] = Field(None, description="User profile")


class MainAppTaxRequest(BaseModel):
    """Request model for main app tax suggestions"""
    session_id: str = Field(..., description="Session ID")
    salary_data: Dict[str, Any] = Field(..., description="Extracted salary data")
    user_info: Dict[str, Any] = Field(..., description="Additional user information")


class MainAppChatRequest(BaseModel):
    """Request model for main app chat"""
    session_id: str = Field(..., description="Session ID")
    message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(None, description="Context data")


class AIResponse(BaseModel):
    """Response model for AI responses"""
    success: bool
    response_type: str
    content: str
    confidence_score: float
    processing_time: float
    tokens_used: int
    suggestions: Optional[List[Dict[str, Any]]] = None
    regime_comparison: Optional[Dict[str, Any]] = None
    investment_plan: Optional[Dict[str, Any]] = None
    action_items: Optional[List[str]] = None
    disclaimers: Optional[List[str]] = None
    error_message: Optional[str] = None
    timestamp: datetime


class ServiceStatus(BaseModel):
    """Service status model"""
    is_available: bool
    model_name: str
    rate_limits: Dict[str, Any]
    configuration: Dict[str, Any]


@router.get("/status", response_model=ServiceStatus)
async def get_service_status():
    """Get AI service status"""
    try:
        status = gemini_service.get_service_status()
        return ServiceStatus(**status)
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get service status")


@router.post("/test-connection")
async def test_connection():
    """Test AI service connection"""
    try:
        result = await gemini_service.test_connection()
        return result
    except Exception as e:
        logger.error(f"Error testing connection: {e}")
        raise HTTPException(status_code=500, detail="Failed to test connection")


@router.post("/tax-suggestions", response_model=AIResponse)
async def generate_tax_suggestions(request: Union[TaxSuggestionsRequest, MainAppTaxRequest]):
    """Generate personalized tax savings suggestions"""
    try:
        # Handle both formats - demo format and main app format
        if hasattr(request, 'salary_data'):
            # Main app format
            salary_data = request.salary_data
            user_info = request.user_info
            
            context = {
                "basic_salary": salary_data.get("basic_salary", 0),
                "hra": salary_data.get("hra", 0),
                "other_allowances": salary_data.get("other_allowances", 0),
                "gross_salary": salary_data.get("gross_salary", 0),
                "current_deductions": salary_data.get("tax_deducted", 0),
                "net_salary": salary_data.get("net_salary", 0),
                "age": user_info.get("age", 30),
                "marital_status": user_info.get("marital_status", "single"),
                "dependents": 0,
                "city": user_info.get("city", "Mumbai"),
                "rent_paid": user_info.get("monthly_rent", 0),
                "current_investments": "Standard investments",
                "financial_goals": "Tax optimization",
                "session_id": request.session_id
            }
        else:
            # Demo format
            context = request.dict()
        
        # Generate prompt
        prompt = prompt_templates.get_tax_suggestions_prompt(context)
        
        # Create AI request
        ai_request = GeminiRequest(
            prompt=prompt,
            response_type=ResponseType.TAX_SUGGESTIONS,
            context=context,
            session_id=context["session_id"]
        )
        
        # Generate response
        response = await gemini_service.generate_response(ai_request)
        
        # Parse response
        parsed_response = parser.parse_response(response.content, ResponseType.TAX_SUGGESTIONS)
        
        return AIResponse(
            success=response.is_valid,
            response_type=response.response_type.value,
            content=response.content,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            tokens_used=response.tokens_used,
            suggestions=[parser.to_dict(s) for s in parsed_response.suggestions] if parsed_response.suggestions else None,
            action_items=parsed_response.action_items,
            disclaimers=parsed_response.disclaimers,
            error_message=response.error_message,
            timestamp=response.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error generating tax suggestions: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate tax suggestions")


@router.post("/regime-comparison", response_model=AIResponse)
async def compare_tax_regimes(request: Union[RegimeComparisonRequest, MainAppTaxRequest]):
    """Compare old vs new tax regimes"""
    try:
        # Handle both formats - demo format and main app format
        if hasattr(request, 'salary_data'):
            # Main app format
            salary_data = request.salary_data
            user_info = request.user_info
            
            context = {
                "gross_salary": salary_data.get("gross_salary", 0),
                "basic_salary": salary_data.get("basic_salary", 0),
                "hra": salary_data.get("hra", 0),
                "other_allowances": salary_data.get("other_allowances", 0),
                "section_80c": 150000,  # Default assumption
                "section_80d": 25000,   # Default assumption
                "home_loan_interest": 0,
                "other_deductions": salary_data.get("tax_deducted", 0),
                "age": user_info.get("age", 30),
                "city": user_info.get("city", "Mumbai"),
                "rent_paid": user_info.get("monthly_rent", 0),
                "session_id": request.session_id
            }
        else:
            # Demo format
            context = request.dict()
        
        # Generate prompt
        prompt = prompt_templates.get_regime_comparison_prompt(context)
        
        # Create AI request
        ai_request = GeminiRequest(
            prompt=prompt,
            response_type=ResponseType.REGIME_COMPARISON,
            context=context,
            session_id=context["session_id"]
        )
        
        # Generate response
        response = await gemini_service.generate_response(ai_request)
        
        # Parse response
        parsed_response = parser.parse_response(response.content, ResponseType.REGIME_COMPARISON)
        
        return AIResponse(
            success=response.is_valid,
            response_type=response.response_type.value,
            content=response.content,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            tokens_used=response.tokens_used,
            regime_comparison=parser.to_dict(parsed_response.regime_comparison) if parsed_response.regime_comparison else None,
            disclaimers=parsed_response.disclaimers,
            error_message=response.error_message,
            timestamp=response.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error comparing regimes: {e}")
        raise HTTPException(status_code=500, detail="Failed to compare tax regimes")


@router.post("/chat", response_model=AIResponse)
async def chat_with_ai(request: Union[ChatRequest, MainAppChatRequest]):
    """Chat with AI tax advisor"""
    try:
        # Handle both formats
        if hasattr(request, 'message'):
            # Main app format
            context = request.context or {}
            context["session_id"] = request.session_id
            user_query = request.message
        else:
            # Demo format
            context = request.dict()
            user_query = request.user_query
        
        # Generate prompt
        prompt = prompt_templates.get_chat_response_prompt(context, user_query)
        
        # Create AI request
        ai_request = GeminiRequest(
            prompt=prompt,
            response_type=ResponseType.CHAT_RESPONSE,
            context=context,
            session_id=context["session_id"]
        )
        
        # Generate response
        response = await gemini_service.generate_response(ai_request)
        
        # Parse response
        parsed_response = parser.parse_response(response.content, ResponseType.CHAT_RESPONSE)
        
        return AIResponse(
            success=response.is_valid,
            response_type=response.response_type.value,
            content=response.content,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            tokens_used=response.tokens_used,
            disclaimers=parsed_response.disclaimers,
            error_message=response.error_message,
            timestamp=response.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="Failed to process chat request")


@router.get("/prompt-templates/{prompt_type}")
async def get_prompt_template(prompt_type: str):
    """Get prompt template for a specific type"""
    try:
        # Convert string to enum
        prompt_type_enum = PromptType(prompt_type)
        
        # Get default context
        context = prompt_templates.get_default_context(prompt_type_enum)
        
        # Generate prompt
        prompt = prompt_templates.get_prompt_by_type(prompt_type_enum, context)
        
        return {
            "prompt_type": prompt_type,
            "prompt": prompt,
            "required_context": prompt_templates.get_required_context(prompt_type_enum),
            "sample_context": context
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prompt type")
    except Exception as e:
        logger.error(f"Error getting prompt template: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prompt template")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "service_available": gemini_service.is_available(),
        "version": "1.0.0"
    } 