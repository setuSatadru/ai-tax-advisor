"""
Prompt Templates for Tax Advisor AI Integration
Contains structured prompts for different tax scenarios
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class PromptType(Enum):
    """Types of prompts for different tax scenarios"""
    TAX_SUGGESTIONS = "tax_suggestions"
    REGIME_COMPARISON = "regime_comparison"
    INVESTMENT_ADVICE = "investment_advice"
    DEDUCTION_OPTIMIZATION = "deduction_optimization"
    CHAT_RESPONSE = "chat_response"
    FIRST_TIME_PLANNING = "first_time_planning"
    QUARTERLY_PLANNING = "quarterly_planning"


@dataclass
class PromptTemplate:
    """Structure for prompt templates"""
    name: str
    template: str
    required_context: List[str]
    optional_context: List[str]
    example_response: str


class TaxPromptTemplates:
    """Collection of tax-related prompt templates"""
    
    # Base context for all tax-related prompts
    BASE_CONTEXT = """
You are an expert Indian tax advisor with deep knowledge of Income Tax Act 1961, current tax rates, and investment options. 
You provide accurate, personalized tax advice based on user's financial situation.

Current Tax Year: FY 2024-25 (AY 2025-26)
Current Tax Rates and Slabs:
- Old Regime: 0% (0-2.5L), 5% (2.5-5L), 20% (5-10L), 30% (10L+)
- New Regime: 0% (0-3L), 5% (3-6L), 10% (6-9L), 15% (9-12L), 20% (12-15L), 30% (15L+)

Important Sections:
- Section 80C: Up to ₹1.5L (ELSS, PPF, NSC, Home Loan Principal, etc.)
- Section 80D: Health Insurance Premium (Self: ₹25K, Parents: ₹25K, Senior Citizens: ₹50K)
- Section 80G: Donations to eligible institutions
- Section 24(b): Home Loan Interest (Self-occupied: ₹2L, Let-out: No limit)
- Section 80E: Education Loan Interest (No limit)

Always provide specific, actionable advice with calculations where possible.
"""
    
    @classmethod
    def get_tax_suggestions_prompt(cls, context: Dict[str, Any]) -> str:
        """Generate comprehensive tax savings suggestions"""
        template = cls.BASE_CONTEXT + """
Based on the following financial information, provide personalized tax-saving suggestions:

SALARY INFORMATION:
- Basic Salary: ₹{basic_salary:,}
- HRA: ₹{hra:,}
- Other Allowances: ₹{other_allowances:,}
- Gross Salary: ₹{gross_salary:,}
- Current Deductions: ₹{current_deductions:,}
- Net Salary: ₹{net_salary:,}

PERSONAL INFORMATION:
- Age: {age} years
- Marital Status: {marital_status}
- Dependents: {dependents}
- City: {city}
- Rent Paid: ₹{rent_paid:,}/month

CURRENT INVESTMENTS:
{current_investments}

FINANCIAL GOALS:
{financial_goals}

INSTRUCTIONS:
1. Analyze the current tax situation for both old and new regime
2. Suggest specific investment options with amounts
3. Identify missed deduction opportunities
4. Provide month-wise investment planning
5. Calculate potential tax savings
6. Prioritize suggestions based on risk and returns

Format your response as:
## Tax Analysis Summary
## Regime Recommendation
## Top 5 Tax-Saving Suggestions
## Investment Allocation Plan
## Monthly Action Plan
## Potential Tax Savings

Be specific with amounts and provide clear reasoning for each suggestion.
"""
        return template.format(**context)
    
    @classmethod
    def get_regime_comparison_prompt(cls, context: Dict[str, Any]) -> str:
        """Compare old vs new tax regime"""
        template = cls.BASE_CONTEXT + """
Perform a detailed comparison between Old and New tax regimes for the following profile:

INCOME DETAILS:
- Gross Annual Income: ₹{gross_salary:,}
- Basic Salary: ₹{basic_salary:,}
- HRA: ₹{hra:,}
- Other Allowances: ₹{other_allowances:,}

CURRENT DEDUCTIONS/INVESTMENTS:
- Section 80C Investments: ₹{section_80c:,}
- Section 80D (Health Insurance): ₹{section_80d:,}
- Home Loan Interest: ₹{home_loan_interest:,}
- Other Deductions: ₹{other_deductions:,}

PERSONAL DETAILS:
- Age: {age}
- City: {city}
- Rent Paid: ₹{rent_paid:,}/month

ANALYSIS REQUIRED:
1. Calculate tax liability under both regimes
2. Consider HRA exemption impact
3. Factor in standard deduction differences
4. Account for all applicable deductions
5. Provide break-even analysis
6. Suggest optimization strategies for chosen regime

Format your response as:
## Old Regime Calculation
## New Regime Calculation  
## Comparison Summary
## Recommendation with Reasoning
## Optimization Tips
## Future Planning Considerations

Show detailed calculations and provide clear recommendation.
"""
        return template.format(**context)
    
    @classmethod
    def get_investment_advice_prompt(cls, context: Dict[str, Any]) -> str:
        """Provide investment advice for tax planning"""
        template = cls.BASE_CONTEXT + """
Provide comprehensive investment advice for tax planning:

INVESTOR PROFILE:
- Age: {age} years
- Annual Income: ₹{annual_income:,}
- Monthly Surplus: ₹{monthly_surplus:,}
- Risk Appetite: {risk_appetite}
- Investment Experience: {investment_experience}
- Investment Horizon: {investment_horizon}

CURRENT PORTFOLIO:
{current_investments}

FINANCIAL GOALS:
{financial_goals}

TAX PLANNING NEEDS:
- Section 80C Gap: ₹{section_80c_gap:,}
- Section 80D Gap: ₹{section_80d_gap:,}
- Other Tax-saving Opportunities: {other_opportunities}

PROVIDE RECOMMENDATIONS FOR:
1. Asset allocation strategy
2. Tax-saving investment options
3. Risk-appropriate product selection
4. SIP vs lump sum recommendations
5. Product-specific suggestions with rationale
6. Timeline and review schedule

Format your response as:
## Investment Strategy Overview
## Asset Allocation Recommendation
## Tax-Saving Investment Plan
## Product Recommendations
## Implementation Timeline
## Review and Monitoring Plan

Focus on tax-efficient investments while maintaining portfolio balance.
"""
        return template.format(**context)
    
    @classmethod
    def get_deduction_optimization_prompt(cls, context: Dict[str, Any]) -> str:
        """Optimize deductions and exemptions"""
        template = cls.BASE_CONTEXT + """
Analyze and optimize deductions and exemptions:

CURRENT DEDUCTIONS:
- Section 80C: ₹{section_80c_current:,} (Limit: ₹1.5L)
- Section 80D: ₹{section_80d_current:,} (Limit varies)
- Section 80G: ₹{section_80g_current:,}
- Section 24(b): ₹{section_24b_current:,}
- Section 80E: ₹{section_80e_current:,}
- Other Deductions: ₹{other_deductions:,}

INCOME DETAILS:
- Gross Salary: ₹{gross_salary:,}
- HRA: ₹{hra:,}
- Rent Paid: ₹{rent_paid:,}/month
- City: {city}

PERSONAL SITUATION:
- Age: {age}
- Dependents: {dependents}
- Health Insurance: {health_insurance_details}
- Home Loan: {home_loan_details}
- Education Loan: {education_loan_details}

OPTIMIZATION ANALYSIS:
1. Identify underutilized deduction sections
2. Calculate HRA exemption optimization
3. Suggest additional deduction opportunities
4. Recommend timing strategies
5. Provide documentation requirements
6. Calculate potential additional savings

Format your response as:
## Current Deduction Analysis
## Missed Opportunities
## Optimization Recommendations
## HRA Exemption Strategy
## Additional Deduction Suggestions
## Implementation Checklist
## Potential Additional Savings

Provide specific amounts and actionable steps.
"""
        return template.format(**context)
    
    @classmethod
    def get_chat_response_prompt(cls, context: Dict[str, Any]) -> str:
        """Handle general tax-related queries"""
        template = cls.BASE_CONTEXT + """
User Context:
- Session ID: {session_id}
- Previous Context: {previous_context}
- User Profile: {user_profile}

User Query: {user_query}

INSTRUCTIONS:
1. Provide accurate, helpful tax advice
2. Reference specific sections of Income Tax Act when applicable
3. Give practical, actionable suggestions
4. Ask clarifying questions if needed
5. Maintain conversation context
6. Provide examples and calculations where relevant

Keep responses conversational yet professional. If the query is outside tax domain, politely redirect to tax-related topics.
"""
        return template.format(**context)
    
    @classmethod
    def get_first_time_planning_prompt(cls, context: Dict[str, Any]) -> str:
        """Guide first-time tax planners"""
        template = cls.BASE_CONTEXT + """
Welcome to tax planning! You're taking a smart step by starting early.

YOUR PROFILE:
- Age: {age} years
- Annual Income: ₹{annual_income:,}
- Employment Type: {employment_type}
- Experience Level: First-time tax planner

CURRENT SITUATION:
- Monthly Savings Capacity: ₹{monthly_savings:,}
- Financial Goals: {financial_goals}
- Risk Tolerance: {risk_tolerance}

BEGINNER-FRIENDLY GUIDANCE NEEDED:
1. Tax planning basics and importance
2. Simple investment options to start with
3. Step-by-step implementation plan
4. Common mistakes to avoid
5. Documentation and record-keeping
6. Timeline for tax planning activities

Format your response as:
## Tax Planning 101
## Your Tax Situation Overview
## Beginner-Friendly Investment Options
## Step-by-Step Action Plan
## Important Deadlines
## Common Mistakes to Avoid
## Next Steps

Use simple language and provide clear, actionable steps for a beginner.
"""
        return template.format(**context)
    
    @classmethod
    def get_quarterly_planning_prompt(cls, context: Dict[str, Any]) -> str:
        """Provide quarterly tax planning guidance"""
        template = cls.BASE_CONTEXT + """
Quarterly Tax Planning Review for Q{quarter} FY{financial_year}:

CURRENT PROGRESS:
- YTD Income: ₹{ytd_income:,}
- YTD Investments: ₹{ytd_investments:,}
- Projected Annual Income: ₹{projected_annual_income:,}
- Remaining Investment Capacity: ₹{remaining_capacity:,}

QUARTERLY GOALS:
- Target Investments for Q{quarter}: ₹{quarterly_target:,}
- Pending Deductions: {pending_deductions}
- Upcoming Expenses: {upcoming_expenses}

QUARTERLY REVIEW ANALYSIS:
1. Progress against annual tax planning goals
2. Adjustments needed based on income changes
3. Optimal timing for remaining investments
4. Quarterly investment recommendations
5. Tax-saving opportunities for this quarter
6. Preparation for next quarter

Format your response as:
## Q{quarter} Progress Review
## Goal Achievement Analysis
## Recommended Actions for This Quarter
## Investment Timing Strategy
## Upcoming Deadlines
## Next Quarter Preparation

Provide specific action items with timelines.
"""
        return template.format(**context)
    
    @classmethod
    def get_prompt_by_type(cls, prompt_type: PromptType, context: Dict[str, Any]) -> str:
        """Get prompt template by type"""
        prompt_map = {
            PromptType.TAX_SUGGESTIONS: cls.get_tax_suggestions_prompt,
            PromptType.REGIME_COMPARISON: cls.get_regime_comparison_prompt,
            PromptType.INVESTMENT_ADVICE: cls.get_investment_advice_prompt,
            PromptType.DEDUCTION_OPTIMIZATION: cls.get_deduction_optimization_prompt,
            PromptType.CHAT_RESPONSE: cls.get_chat_response_prompt,
            PromptType.FIRST_TIME_PLANNING: cls.get_first_time_planning_prompt,
            PromptType.QUARTERLY_PLANNING: cls.get_quarterly_planning_prompt,
        }
        
        if prompt_type not in prompt_map:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        return prompt_map[prompt_type](context)
    
    @classmethod
    def get_required_context(cls, prompt_type: PromptType) -> List[str]:
        """Get required context fields for a prompt type"""
        context_map = {
            PromptType.TAX_SUGGESTIONS: [
                'basic_salary', 'hra', 'other_allowances', 'gross_salary',
                'current_deductions', 'net_salary', 'age', 'marital_status',
                'dependents', 'city', 'rent_paid', 'current_investments', 'financial_goals'
            ],
            PromptType.REGIME_COMPARISON: [
                'gross_salary', 'basic_salary', 'hra', 'other_allowances',
                'section_80c', 'section_80d', 'home_loan_interest', 'other_deductions',
                'age', 'city', 'rent_paid'
            ],
            PromptType.INVESTMENT_ADVICE: [
                'age', 'annual_income', 'monthly_surplus', 'risk_appetite',
                'investment_experience', 'investment_horizon', 'current_investments',
                'financial_goals', 'section_80c_gap', 'section_80d_gap', 'other_opportunities'
            ],
            PromptType.DEDUCTION_OPTIMIZATION: [
                'section_80c_current', 'section_80d_current', 'section_80g_current',
                'section_24b_current', 'section_80e_current', 'other_deductions',
                'gross_salary', 'hra', 'rent_paid', 'city', 'age', 'dependents',
                'health_insurance_details', 'home_loan_details', 'education_loan_details'
            ],
            PromptType.CHAT_RESPONSE: [
                'session_id', 'user_query', 'previous_context', 'user_profile'
            ],
            PromptType.FIRST_TIME_PLANNING: [
                'age', 'annual_income', 'employment_type', 'monthly_savings',
                'financial_goals', 'risk_tolerance'
            ],
            PromptType.QUARTERLY_PLANNING: [
                'quarter', 'financial_year', 'ytd_income', 'ytd_investments',
                'projected_annual_income', 'remaining_capacity', 'quarterly_target',
                'pending_deductions', 'upcoming_expenses'
            ]
        }
        
        return context_map.get(prompt_type, [])
    
    @classmethod
    def validate_context(cls, prompt_type: PromptType, context: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate if context has all required fields"""
        required_fields = cls.get_required_context(prompt_type)
        missing_fields = [field for field in required_fields if field not in context]
        
        return len(missing_fields) == 0, missing_fields
    
    @classmethod
    def get_default_context(cls, prompt_type: PromptType) -> Dict[str, Any]:
        """Get default context values for testing"""
        defaults = {
            'basic_salary': 600000,
            'hra': 240000,
            'other_allowances': 60000,
            'gross_salary': 900000,
            'current_deductions': 50000,
            'net_salary': 700000,
            'age': 30,
            'marital_status': 'Single',
            'dependents': 0,
            'city': 'Mumbai',
            'rent_paid': 25000,
            'current_investments': 'PPF: ₹50,000, ELSS: ₹30,000',
            'financial_goals': 'Home purchase in 5 years, Retirement planning',
            'section_80c': 80000,
            'section_80d': 15000,
            'home_loan_interest': 0,
            'other_deductions': 0,
            'annual_income': 900000,
            'monthly_surplus': 30000,
            'risk_appetite': 'Moderate',
            'investment_experience': 'Beginner',
            'investment_horizon': '5-10 years',
            'section_80c_gap': 70000,
            'section_80d_gap': 10000,
            'other_opportunities': 'None identified',
            'session_id': 'test_session',
            'user_query': 'How can I save more tax?',
            'previous_context': 'None',
            'user_profile': 'Software Engineer, Age 30',
            'employment_type': 'Salaried',
            'monthly_savings': 25000,
            'risk_tolerance': 'Medium',
            'quarter': 2,
            'financial_year': '2024-25',
            'ytd_income': 450000,
            'ytd_investments': 40000,
            'projected_annual_income': 900000,
            'remaining_capacity': 110000,
            'quarterly_target': 40000,
            'pending_deductions': 'Section 80C completion',
            'upcoming_expenses': 'Insurance premium renewal'
        }
        
        required_fields = cls.get_required_context(prompt_type)
        return {field: defaults.get(field, '') for field in required_fields} 