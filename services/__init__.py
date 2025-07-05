"""
Services package for Tax Advisor Application
Contains AI and external service integrations
"""

from .gemini_service import GeminiService, GeminiRequest, GeminiResponse, ResponseType
from .prompt_templates import TaxPromptTemplates, PromptType
from .response_parser import (
    ResponseParser, ParsedResponse, TaxSuggestion, RegimeComparison, 
    InvestmentPlan, SuggestionCategory, SuggestionPriority, response_parser
)

__all__ = [
    "GeminiService", 
    "GeminiRequest", 
    "GeminiResponse", 
    "ResponseType",
    "TaxPromptTemplates", 
    "PromptType",
    "ResponseParser",
    "ParsedResponse",
    "TaxSuggestion",
    "RegimeComparison",
    "InvestmentPlan",
    "SuggestionCategory",
    "SuggestionPriority",
    "response_parser"
] 