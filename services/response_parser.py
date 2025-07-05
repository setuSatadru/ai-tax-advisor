"""
Response Parser and Validator for Tax Advisor AI Integration
Handles parsing and validation of AI responses from Gemini Flash 2.0 Pro
"""

import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging

from .gemini_service import ResponseType

# Configure logging
logger = logging.getLogger(__name__)


class SuggestionPriority(Enum):
    """Priority levels for tax suggestions"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SuggestionCategory(Enum):
    """Categories of tax suggestions"""
    INVESTMENT = "investment"
    DEDUCTION = "deduction"
    REGIME_CHANGE = "regime_change"
    TIMING = "timing"
    DOCUMENTATION = "documentation"
    PLANNING = "planning"


@dataclass
class TaxSuggestion:
    """Structure for individual tax suggestions"""
    title: str
    description: str
    category: SuggestionCategory
    priority: SuggestionPriority
    potential_savings: float
    investment_amount: Optional[float] = None
    implementation_steps: List[str] = None
    timeline: Optional[str] = None
    risk_level: Optional[str] = None
    additional_info: Optional[str] = None


@dataclass
class RegimeComparison:
    """Structure for tax regime comparison"""
    old_regime_tax: float
    new_regime_tax: float
    recommended_regime: str
    savings_difference: float
    key_factors: List[str]
    break_even_point: Optional[str] = None
    optimization_tips: List[str] = None


@dataclass
class InvestmentPlan:
    """Structure for investment recommendations"""
    total_amount: float
    monthly_amount: float
    asset_allocation: Dict[str, float]
    specific_products: List[Dict[str, Any]]
    timeline: str
    review_frequency: str
    risk_assessment: str


@dataclass
class ParsedResponse:
    """Structure for parsed AI response"""
    response_type: ResponseType
    is_valid: bool
    confidence_score: float
    suggestions: List[TaxSuggestion] = None
    regime_comparison: Optional[RegimeComparison] = None
    investment_plan: Optional[InvestmentPlan] = None
    summary: Optional[str] = None
    action_items: List[str] = None
    disclaimers: List[str] = None
    error_messages: List[str] = None
    raw_content: Optional[str] = None


class ResponseParser:
    """Parser for AI responses from Gemini Flash 2.0 Pro"""
    
    def __init__(self):
        """Initialize response parser"""
        self.section_patterns = {
            'tax_analysis': r'##\s*Tax Analysis Summary\s*\n(.*?)(?=##|\Z)',
            'regime_recommendation': r'##\s*Regime Recommendation\s*\n(.*?)(?=##|\Z)',
            'suggestions': r'##\s*(?:Top \d+ )?Tax[-\s]*Saving Suggestions\s*\n(.*?)(?=##|\Z)',
            'investment_plan': r'##\s*Investment (?:Allocation )?Plan\s*\n(.*?)(?=##|\Z)',
            'monthly_plan': r'##\s*Monthly Action Plan\s*\n(.*?)(?=##|\Z)',
            'potential_savings': r'##\s*Potential Tax Savings\s*\n(.*?)(?=##|\Z)',
            'old_regime': r'##\s*Old Regime Calculation\s*\n(.*?)(?=##|\Z)',
            'new_regime': r'##\s*New Regime Calculation\s*\n(.*?)(?=##|\Z)',
            'comparison': r'##\s*Comparison Summary\s*\n(.*?)(?=##|\Z)',
            'optimization': r'##\s*Optimization (?:Tips|Recommendations)\s*\n(.*?)(?=##|\Z)'
        }
        
        self.amount_pattern = r'₹\s*([0-9,]+(?:\.[0-9]+)?)'
        self.percentage_pattern = r'(\d+(?:\.\d+)?)\s*%'
        
    def parse_response(self, content: str, response_type: ResponseType) -> ParsedResponse:
        """Parse AI response based on type"""
        try:
            if response_type == ResponseType.TAX_SUGGESTIONS:
                return self._parse_tax_suggestions(content)
            elif response_type == ResponseType.REGIME_COMPARISON:
                return self._parse_regime_comparison(content)
            elif response_type == ResponseType.INVESTMENT_ADVICE:
                return self._parse_investment_advice(content)
            elif response_type == ResponseType.DEDUCTION_OPTIMIZATION:
                return self._parse_deduction_optimization(content)
            elif response_type == ResponseType.CHAT_RESPONSE:
                return self._parse_chat_response(content)
            else:
                return self._parse_generic_response(content, response_type)
                
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return ParsedResponse(
                response_type=response_type,
                is_valid=False,
                confidence_score=0.0,
                error_messages=[f"Parsing error: {str(e)}"],
                raw_content=content
            )
    
    def _parse_tax_suggestions(self, content: str) -> ParsedResponse:
        """Parse tax suggestions response"""
        sections = self._extract_sections(content)
        suggestions = []
        
        # Extract suggestions from the suggestions section
        if 'suggestions' in sections:
            suggestions = self._extract_suggestions_from_text(sections['suggestions'])
        
        # Extract potential savings
        potential_savings = 0.0
        if 'potential_savings' in sections:
            potential_savings = self._extract_amount(sections['potential_savings'])
        
        # Extract action items
        action_items = []
        if 'monthly_plan' in sections:
            action_items = self._extract_action_items(sections['monthly_plan'])
        
        return ParsedResponse(
            response_type=ResponseType.TAX_SUGGESTIONS,
            is_valid=len(suggestions) > 0,
            confidence_score=self._calculate_parsing_confidence(sections),
            suggestions=suggestions,
            summary=sections.get('tax_analysis', ''),
            action_items=action_items,
            disclaimers=self._extract_disclaimers(content),
            raw_content=content
        )
    
    def _parse_regime_comparison(self, content: str) -> ParsedResponse:
        """Parse regime comparison response"""
        sections = self._extract_sections(content)
        
        # Extract tax amounts
        old_regime_tax = 0.0
        new_regime_tax = 0.0
        
        if 'old_regime' in sections:
            old_regime_tax = self._extract_amount(sections['old_regime'])
        
        if 'new_regime' in sections:
            new_regime_tax = self._extract_amount(sections['new_regime'])
        
        # Determine recommended regime
        recommended_regime = "Old Regime"
        if new_regime_tax < old_regime_tax:
            recommended_regime = "New Regime"
        
        # Extract key factors
        key_factors = []
        if 'comparison' in sections:
            key_factors = self._extract_bullet_points(sections['comparison'])
        
        regime_comparison = RegimeComparison(
            old_regime_tax=old_regime_tax,
            new_regime_tax=new_regime_tax,
            recommended_regime=recommended_regime,
            savings_difference=abs(old_regime_tax - new_regime_tax),
            key_factors=key_factors,
            optimization_tips=self._extract_bullet_points(sections.get('optimization', ''))
        )
        
        return ParsedResponse(
            response_type=ResponseType.REGIME_COMPARISON,
            is_valid=old_regime_tax > 0 or new_regime_tax > 0,
            confidence_score=self._calculate_parsing_confidence(sections),
            regime_comparison=regime_comparison,
            summary=sections.get('comparison', ''),
            disclaimers=self._extract_disclaimers(content),
            raw_content=content
        )
    
    def _parse_investment_advice(self, content: str) -> ParsedResponse:
        """Parse investment advice response"""
        sections = self._extract_sections(content)
        
        # Extract investment plan details
        investment_plan = None
        if 'investment_plan' in sections:
            investment_plan = self._extract_investment_plan(sections['investment_plan'])
        
        # Extract suggestions
        suggestions = []
        if 'suggestions' in sections:
            suggestions = self._extract_suggestions_from_text(sections['suggestions'])
        
        return ParsedResponse(
            response_type=ResponseType.INVESTMENT_ADVICE,
            is_valid=investment_plan is not None or len(suggestions) > 0,
            confidence_score=self._calculate_parsing_confidence(sections),
            suggestions=suggestions,
            investment_plan=investment_plan,
            summary=sections.get('tax_analysis', ''),
            disclaimers=self._extract_disclaimers(content),
            raw_content=content
        )
    
    def _parse_deduction_optimization(self, content: str) -> ParsedResponse:
        """Parse deduction optimization response"""
        sections = self._extract_sections(content)
        
        # Extract optimization suggestions
        suggestions = []
        if 'optimization' in sections:
            suggestions = self._extract_suggestions_from_text(sections['optimization'])
        
        # Extract action items
        action_items = []
        if 'monthly_plan' in sections:
            action_items = self._extract_action_items(sections['monthly_plan'])
        
        return ParsedResponse(
            response_type=ResponseType.DEDUCTION_OPTIMIZATION,
            is_valid=len(suggestions) > 0,
            confidence_score=self._calculate_parsing_confidence(sections),
            suggestions=suggestions,
            action_items=action_items,
            summary=sections.get('tax_analysis', ''),
            disclaimers=self._extract_disclaimers(content),
            raw_content=content
        )
    
    def _parse_chat_response(self, content: str) -> ParsedResponse:
        """Parse chat response"""
        # For chat responses, we keep it simple
        return ParsedResponse(
            response_type=ResponseType.CHAT_RESPONSE,
            is_valid=len(content.strip()) > 0,
            confidence_score=0.8,  # Default confidence for chat
            summary=content,
            disclaimers=self._extract_disclaimers(content),
            raw_content=content
        )
    
    def _parse_generic_response(self, content: str, response_type: ResponseType) -> ParsedResponse:
        """Parse generic response"""
        sections = self._extract_sections(content)
        
        return ParsedResponse(
            response_type=response_type,
            is_valid=len(content.strip()) > 0,
            confidence_score=self._calculate_parsing_confidence(sections),
            summary=content,
            disclaimers=self._extract_disclaimers(content),
            raw_content=content
        )
    
    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from structured response"""
        sections = {}
        
        for section_name, pattern in self.section_patterns.items():
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                sections[section_name] = matches[0].strip()
        
        return sections
    
    def _extract_suggestions_from_text(self, text: str) -> List[TaxSuggestion]:
        """Extract tax suggestions from text"""
        suggestions = []
        
        # Look for numbered or bulleted lists
        suggestion_patterns = [
            r'(\d+)\.\s*([^\n]+)\n(.*?)(?=\d+\.|$)',
            r'[•\-\*]\s*([^\n]+)\n(.*?)(?=[•\-\*]|$)',
        ]
        
        for pattern in suggestion_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if len(match) >= 2:
                    title = match[1] if len(match) > 2 else match[0]
                    description = match[2] if len(match) > 2 else match[1]
                    
                    # Extract potential savings
                    potential_savings = self._extract_amount(description)
                    
                    # Determine category and priority
                    category = self._determine_category(title + " " + description)
                    priority = self._determine_priority(description)
                    
                    suggestions.append(TaxSuggestion(
                        title=title.strip(),
                        description=description.strip(),
                        category=category,
                        priority=priority,
                        potential_savings=potential_savings,
                        implementation_steps=self._extract_bullet_points(description)
                    ))
        
        return suggestions
    
    def _extract_investment_plan(self, text: str) -> InvestmentPlan:
        """Extract investment plan from text"""
        total_amount = self._extract_amount(text)
        monthly_amount = total_amount / 12 if total_amount > 0 else 0
        
        # Extract asset allocation (simplified)
        asset_allocation = {}
        allocation_pattern = r'(\w+(?:\s+\w+)?)\s*:\s*(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(allocation_pattern, text, re.IGNORECASE)
        
        for match in matches:
            asset_type = match[0].strip()
            percentage = float(match[1])
            asset_allocation[asset_type] = percentage
        
        return InvestmentPlan(
            total_amount=total_amount,
            monthly_amount=monthly_amount,
            asset_allocation=asset_allocation,
            specific_products=[],
            timeline="12 months",
            review_frequency="Quarterly",
            risk_assessment="Medium"
        )
    
    def _extract_amount(self, text: str) -> float:
        """Extract monetary amount from text"""
        matches = re.findall(self.amount_pattern, text)
        if matches:
            # Take the largest amount found
            amounts = [float(match.replace(',', '')) for match in matches]
            return max(amounts)
        return 0.0
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text"""
        bullet_patterns = [
            r'[•\-\*]\s*([^\n]+)',
            r'(\d+)\.\s*([^\n]+)',
        ]
        
        points = []
        for pattern in bullet_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                point = match[1] if isinstance(match, tuple) and len(match) > 1 else match[0] if isinstance(match, tuple) else match
                points.append(point.strip())
        
        return points
    
    def _extract_action_items(self, text: str) -> List[str]:
        """Extract action items from text"""
        return self._extract_bullet_points(text)
    
    def _extract_disclaimers(self, content: str) -> List[str]:
        """Extract disclaimers from content"""
        disclaimers = []
        
        # Common disclaimer patterns
        disclaimer_patterns = [
            r'(?:disclaimer|note|important|warning)[:\s]*(.*?)(?=\n\n|\Z)',
            r'\*\*(?:disclaimer|note|important|warning)\*\*[:\s]*(.*?)(?=\n\n|\Z)',
        ]
        
        for pattern in disclaimer_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
            disclaimers.extend([match.strip() for match in matches if match.strip()])
        
        # Add default disclaimer if none found
        if not disclaimers:
            disclaimers.append("This advice is for informational purposes only. Please consult with a qualified tax advisor before making investment decisions.")
        
        return disclaimers
    
    def _determine_category(self, text: str) -> SuggestionCategory:
        """Determine suggestion category from text"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['invest', 'elss', 'ppf', 'nsc', 'mutual fund']):
            return SuggestionCategory.INVESTMENT
        elif any(keyword in text_lower for keyword in ['deduction', '80c', '80d', 'exemption']):
            return SuggestionCategory.DEDUCTION
        elif any(keyword in text_lower for keyword in ['regime', 'old', 'new']):
            return SuggestionCategory.REGIME_CHANGE
        elif any(keyword in text_lower for keyword in ['timing', 'when', 'deadline']):
            return SuggestionCategory.TIMING
        elif any(keyword in text_lower for keyword in ['document', 'proof', 'receipt']):
            return SuggestionCategory.DOCUMENTATION
        else:
            return SuggestionCategory.PLANNING
    
    def _determine_priority(self, text: str) -> SuggestionPriority:
        """Determine suggestion priority from text"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['urgent', 'immediate', 'critical', 'high']):
            return SuggestionPriority.HIGH
        elif any(keyword in text_lower for keyword in ['medium', 'moderate', 'consider']):
            return SuggestionPriority.MEDIUM
        else:
            return SuggestionPriority.LOW
    
    def _calculate_parsing_confidence(self, sections: Dict[str, str]) -> float:
        """Calculate confidence score for parsing"""
        base_score = 0.5
        
        # Add points for each section found
        section_score = len(sections) * 0.1
        
        # Add points for structured content
        structure_score = 0.0
        for section_content in sections.values():
            if re.search(r'##|\*\*|\d+\.', section_content):
                structure_score += 0.05
        
        confidence = min(base_score + section_score + structure_score, 1.0)
        return round(confidence, 2)
    
    def validate_parsed_response(self, parsed_response: ParsedResponse) -> Tuple[bool, List[str]]:
        """Validate parsed response"""
        errors = []
        
        if not parsed_response.is_valid:
            errors.append("Response marked as invalid")
        
        if parsed_response.confidence_score < 0.3:
            errors.append("Low confidence score in parsing")
        
        if parsed_response.response_type == ResponseType.TAX_SUGGESTIONS:
            if not parsed_response.suggestions or len(parsed_response.suggestions) == 0:
                errors.append("No tax suggestions found")
        
        elif parsed_response.response_type == ResponseType.REGIME_COMPARISON:
            if not parsed_response.regime_comparison:
                errors.append("No regime comparison data found")
        
        elif parsed_response.response_type == ResponseType.INVESTMENT_ADVICE:
            if not parsed_response.investment_plan and not parsed_response.suggestions:
                errors.append("No investment advice found")
        
        return len(errors) == 0, errors
    
    def to_dict(self, parsed_response: ParsedResponse) -> Dict[str, Any]:
        """Convert parsed response to dictionary"""
        return asdict(parsed_response)
    
    def to_json(self, parsed_response: ParsedResponse) -> str:
        """Convert parsed response to JSON"""
        return json.dumps(self.to_dict(parsed_response), indent=2, default=str)


# Global parser instance
response_parser = ResponseParser() 