import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

# Configure Streamlit page
st.set_page_config(
    page_title="AI Pension Benefits Optimizer",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class UserProfile:
    age: int
    current_salary: float
    years_of_service: int
    expected_retirement_age: int
    marital_status: str
    dependents: int
    location: str
    pension_plan_type: str
    current_pension_balance: float
    monthly_contributions: float
    risk_tolerance: str
    health_status: str
    other_savings: float
    spouse_pension: float

class PensionOptimizer:
    def __init__(self):
        self.inflation_rate = 0.06  # 6% average inflation
        self.market_return = 0.10   # 10% average market return
        self.safe_return = 0.07     # 7% safe investment return
        self.tax_rates = {
            'low': 0.10,
            'medium': 0.20,
            'high': 0.30
        }
        
    def calculate_life_expectancy(self, age: int, gender: str, health: str) -> int:
        """Calculate life expectancy based on demographics and health"""
        base_expectancy = 78 if gender == 'Male' else 82
        health_adjustment = {'Excellent': 5, 'Good': 2, 'Fair': 0, 'Poor': -3}
        return base_expectancy + health_adjustment.get(health, 0)
    
    def project_pension_growth(self, current_balance: float, monthly_contrib: float, 
                             years_to_retirement: int, return_rate: float) -> float:
        """Project pension balance at retirement"""
        months = years_to_retirement * 12
        monthly_rate = return_rate / 12
        
        # Future value of current balance
        fv_current = current_balance * (1 + return_rate) ** years_to_retirement
        
        # Future value of monthly contributions
        if monthly_rate > 0:
            fv_contributions = monthly_contrib * (((1 + monthly_rate) ** months - 1) / monthly_rate)
        else:
            fv_contributions = monthly_contrib * months
            
        return fv_current + fv_contributions
    
    def calculate_annuity_payment(self, principal: float, years: int, return_rate: float) -> float:
        """Calculate annual annuity payment"""
        if return_rate == 0:
            return principal / years
        return principal * (return_rate * (1 + return_rate) ** years) / ((1 + return_rate) ** years - 1)
    
    def simulate_lump_sum_strategy(self, profile: UserProfile) -> Dict:
        """Simulate lump sum withdrawal strategy"""
        years_to_retirement = profile.expected_retirement_age - profile.age
        life_expectancy = self.calculate_life_expectancy(profile.age, 'Male', profile.health_status)
        retirement_years = life_expectancy - profile.expected_retirement_age
        
        # Project pension balance at retirement
        pension_at_retirement = self.project_pension_growth(
            profile.current_pension_balance,
            profile.monthly_contributions,
            years_to_retirement,
            self.market_return
        )
        
        # Calculate tax on lump sum (typically higher tax rate)
        tax_rate = self.tax_rates['high']  # Lump sum usually taxed at higher rate
        after_tax_amount = pension_at_retirement * (1 - tax_rate)
        
        # Invest the after-tax amount
        annual_withdrawal = self.calculate_annuity_payment(
            after_tax_amount, retirement_years, self.safe_return
        )
        
        # Adjust for inflation
        real_annual_income = annual_withdrawal / ((1 + self.inflation_rate) ** years_to_retirement)
        
        return {
            'strategy': 'Lump Sum',
            'pension_at_retirement': pension_at_retirement,
            'after_tax_amount': after_tax_amount,
            'annual_income': annual_withdrawal,
            'real_annual_income': real_annual_income,
            'total_lifetime_income': annual_withdrawal * retirement_years,
            'tax_impact': pension_at_retirement - after_tax_amount,
            'pros': ['Full control over investments', 'Potential for higher returns', 'Inheritance benefit'],
            'cons': ['Market risk', 'Inflation risk', 'High upfront tax burden', 'Longevity risk']
        }
    
    def simulate_annuity_strategy(self, profile: UserProfile) -> Dict:
        """Simulate annuity strategy"""
        years_to_retirement = profile.expected_retirement_age - profile.age
        life_expectancy = self.calculate_life_expectancy(profile.age, 'Male', profile.health_status)
        retirement_years = life_expectancy - profile.expected_retirement_age
        
        # Project pension balance at retirement
        pension_at_retirement = self.project_pension_growth(
            profile.current_pension_balance,
            profile.monthly_contributions,
            years_to_retirement,
            self.market_return
        )
        
        # Convert to annuity (lower tax rate, spread over years)
        tax_rate = self.tax_rates['medium']
        annual_gross_payment = self.calculate_annuity_payment(
            pension_at_retirement, retirement_years, 0.04  # Conservative annuity rate
        )
        annual_after_tax = annual_gross_payment * (1 - tax_rate)
        
        # Adjust for inflation
        real_annual_income = annual_after_tax / ((1 + self.inflation_rate) ** years_to_retirement)
        
        return {
            'strategy': 'Annuity',
            'pension_at_retirement': pension_at_retirement,
            'annual_gross_payment': annual_gross_payment,
            'annual_income': annual_after_tax,
            'real_annual_income': real_annual_income,
            'total_lifetime_income': annual_after_tax * retirement_years,
            'tax_impact': annual_gross_payment * tax_rate * retirement_years,
            'pros': ['Guaranteed income for life', 'Protection from market volatility', 'Lower tax rate'],
            'cons': ['No inflation protection', 'No inheritance benefit', 'Lower potential returns']
        }
    
    def simulate_phased_withdrawal_strategy(self, profile: UserProfile) -> Dict:
        """Simulate phased withdrawal strategy"""
        years_to_retirement = profile.expected_retirement_age - profile.age
        life_expectancy = self.calculate_life_expectancy(profile.age, 'Male', profile.health_status)
        retirement_years = life_expectancy - profile.expected_retirement_age
        
        # Project pension balance at retirement
        pension_at_retirement = self.project_pension_growth(
            profile.current_pension_balance,
            profile.monthly_contributions,
            years_to_retirement,
            self.market_return
        )
        
        # Phased withdrawal (4% rule with adjustments)
        annual_withdrawal_rate = 0.045  # Slightly higher than 4% rule
        annual_gross_withdrawal = pension_at_retirement * annual_withdrawal_rate
        
        # Tax rate varies by withdrawal amount
        tax_rate = self.tax_rates['low']  # Lower tax rate due to controlled withdrawals
        annual_after_tax = annual_gross_withdrawal * (1 - tax_rate)
        
        # Adjust for inflation
        real_annual_income = annual_after_tax / ((1 + self.inflation_rate) ** years_to_retirement)
        
        # Total lifetime income (assuming balance grows)
        total_lifetime = annual_after_tax * retirement_years * 1.1  # Growth factor
        
        return {
            'strategy': 'Phased Withdrawal',
            'pension_at_retirement': pension_at_retirement,
            'annual_withdrawal_rate': annual_withdrawal_rate,
            'annual_income': annual_after_tax,
            'real_annual_income': real_annual_income,
            'total_lifetime_income': total_lifetime,
            'tax_impact': annual_gross_withdrawal * tax_rate * retirement_years,
            'pros': ['Flexible withdrawals', 'Continued growth potential', 'Lower tax rates', 'Inheritance benefit'],
            'cons': ['Market risk', 'Requires active management', 'Longevity risk']
        }
    
    def simulate_hybrid_strategy(self, profile: UserProfile) -> Dict:
        """Simulate hybrid strategy (part annuity, part investment)"""
        years_to_retirement = profile.expected_retirement_age - profile.age
        life_expectancy = self.calculate_life_expectancy(profile.age, 'Male', profile.health_status)
        retirement_years = life_expectancy - profile.expected_retirement_age
        
        # Project pension balance at retirement
        pension_at_retirement = self.project_pension_growth(
            profile.current_pension_balance,
            profile.monthly_contributions,
            years_to_retirement,
            self.market_return
        )
        
        # Split 60% annuity, 40% investment
        annuity_portion = pension_at_retirement * 0.6
        investment_portion = pension_at_retirement * 0.4
        
        # Annuity income
        annuity_annual = self.calculate_annuity_payment(annuity_portion, retirement_years, 0.04)
        annuity_after_tax = annuity_annual * (1 - self.tax_rates['medium'])
        
        # Investment income (phased withdrawal)
        investment_annual = investment_portion * 0.045
        investment_after_tax = investment_annual * (1 - self.tax_rates['low'])
        
        total_annual = annuity_after_tax + investment_after_tax
        real_annual_income = total_annual / ((1 + self.inflation_rate) ** years_to_retirement)
        
        return {
            'strategy': 'Hybrid (60% Annuity + 40% Investment)',
            'pension_at_retirement': pension_at_retirement,
            'annuity_portion': annuity_portion,
            'investment_portion': investment_portion,
            'annual_income': total_annual,
            'real_annual_income': real_annual_income,
            'total_lifetime_income': total_annual * retirement_years * 1.05,
            'tax_impact': (annuity_annual * self.tax_rates['medium'] + investment_annual * self.tax_rates['low']) * retirement_years,
            'pros': ['Balanced risk', 'Some guaranteed income', 'Growth potential', 'Partial inheritance'],
            'cons': ['Complexity', 'Moderate market risk', 'Requires monitoring']
        }
    
    def get_ai_recommendation(self, profile: UserProfile, strategies: List[Dict]) -> Dict:
        """AI-powered recommendation based on user profile"""
        scores = {}
        
        for strategy in strategies:
            score = 0
            strategy_name = strategy['strategy']
            
            # Base score on total lifetime income
            score += strategy['total_lifetime_income'] / 1000000  # Scale down
            
            # Adjust based on risk tolerance
            if profile.risk_tolerance == 'Conservative':
                if 'Annuity' in strategy_name:
                    score += 20
                elif 'Hybrid' in strategy_name:
                    score += 10
                elif 'Lump Sum' in strategy_name:
                    score -= 10
            elif profile.risk_tolerance == 'Aggressive':
                if 'Lump Sum' in strategy_name:
                    score += 20
                elif 'Phased' in strategy_name:
                    score += 15
                elif 'Annuity' in strategy_name:
                    score -= 5
            else:  # Moderate
                if 'Hybrid' in strategy_name:
                    score += 15
                elif 'Phased' in strategy_name:
                    score += 10
            
            # Adjust based on health status
            if profile.health_status in ['Excellent', 'Good']:
                if 'Lump Sum' in strategy_name or 'Phased' in strategy_name:
                    score += 10
            else:
                if 'Annuity' in strategy_name:
                    score += 15
            
            # Adjust based on dependents
            if profile.dependents > 0:
                if 'Lump Sum' in strategy_name or 'Phased' in strategy_name:
                    score += 5
            
            # Adjust based on other savings
            if profile.other_savings > profile.current_salary * 5:
                if 'Annuity' in strategy_name:
                    score += 10
            else:
                if 'Phased' in strategy_name or 'Hybrid' in strategy_name:
                    score += 5
            
            scores[strategy_name] = score
        
        # Find best strategy
        best_strategy = max(scores, key=scores.get)
        confidence = min(95, max(60, scores[best_strategy] * 2))  # Convert to percentage
        
        return {
            'recommended_strategy': best_strategy,
            'confidence_score': confidence,
            'reasoning': self.get_recommendation_reasoning(profile, best_strategy),
            'scores': scores
        }
    
    def get_recommendation_reasoning(self, profile: UserProfile, strategy: str) -> str:
        """Generate reasoning for the recommendation"""
        reasons = []
        
        if 'Annuity' in strategy:
            reasons.append(f"Annuity provides guaranteed income security")
            if profile.health_status in ['Fair', 'Poor']:
                reasons.append("Given health concerns, guaranteed income reduces longevity risk")
            if profile.risk_tolerance == 'Conservative':
                reasons.append("Matches your conservative risk profile")
        
        elif 'Lump Sum' in strategy:
            reasons.append("Maximizes potential returns for aggressive investors")
            if profile.health_status in ['Excellent', 'Good']:
                reasons.append("Good health suggests longer life expectancy, maximizing growth potential")
            if profile.risk_tolerance == 'Aggressive':
                reasons.append("Aligns with your aggressive investment approach")
        
        elif 'Phased' in strategy:
            reasons.append("Offers flexibility while maintaining growth potential")
            if profile.risk_tolerance == 'Moderate':
                reasons.append("Balanced approach suitable for moderate risk tolerance")
        
        elif 'Hybrid' in strategy:
            reasons.append("Provides optimal balance of security and growth")
            reasons.append("Reduces overall risk while maintaining upside potential")
        
        if profile.dependents > 0:
            reasons.append(f"Considers inheritance needs for {profile.dependents} dependent(s)")
        
        if profile.other_savings > profile.current_salary * 3:
            reasons.append("Substantial other savings provide additional security buffer")
        
        return " | ".join(reasons)

def main():
    st.markdown('<h1 class="main-header">🎯 AI Pension Benefits Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("**Maximize your retirement income with AI-powered pension optimization**")
    
    optimizer = PensionOptimizer()
    
    # Sidebar for user input
    with st.sidebar:
        st.header("👤 Your Profile")
        
        # Personal Information
        st.subheader("Personal Details")
        age = st.slider("Current Age", 25, 65, 45)
        expected_retirement_age = st.slider("Expected Retirement Age", age + 1, 75, 60)
        current_salary = st.number_input("Annual Salary (₹)", min_value=100000, max_value=10000000, value=1200000, step=50000)
        years_of_service = st.slider("Years of Service", 0, 40, 15)
        
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        dependents = st.slider("Number of Dependents", 0, 5, 2)
        location = st.selectbox("Location", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Other"])
        
        # Financial Information
        st.subheader("Financial Details")
        current_pension_balance = st.number_input("Current Pension Balance (₹)", min_value=0, max_value=50000000, value=2500000, step=100000)
        monthly_contributions = st.number_input("Monthly Contributions (₹)", min_value=0, max_value=100000, value=15000, step=1000)
        other_savings = st.number_input("Other Savings/Investments (₹)", min_value=0, max_value=50000000, value=1500000, step=100000)
        spouse_pension = st.number_input("Spouse's Pension Value (₹)", min_value=0, max_value=10000000, value=500000, step=50000)
        
        # Preferences
        st.subheader("Preferences & Health")
        pension_plan_type = st.selectbox("Pension Plan Type", ["Employee Provident Fund (EPF)", "National Pension System (NPS)", "Corporate Pension", "Government Pension"])
        risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
        health_status = st.selectbox("Health Status", ["Excellent", "Good", "Fair", "Poor"])
    
    # Create user profile
    profile = UserProfile(
        age=age,
        current_salary=current_salary,
        years_of_service=years_of_service,
        expected_retirement_age=expected_retirement_age,
        marital_status=marital_status,
        dependents=dependents,
        location=location,
        pension_plan_type=pension_plan_type,
        current_pension_balance=current_pension_balance,
        monthly_contributions=monthly_contributions,
        risk_tolerance=risk_tolerance,
        health_status=health_status,
        other_savings=other_savings,
        spouse_pension=spouse_pension
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Strategy Comparison", "🎯 AI Recommendation", "📈 Projections", "🔍 What-If Analysis", "📋 Government Schemes"])
    
    with tab1:
        st.header("Strategy Comparison")
        
        # Calculate all strategies
        with st.spinner("Analyzing pension strategies..."):
            lump_sum = optimizer.simulate_lump_sum_strategy(profile)
            annuity = optimizer.simulate_annuity_strategy(profile)
            phased = optimizer.simulate_phased_withdrawal_strategy(profile)
            hybrid = optimizer.simulate_hybrid_strategy(profile)
            
            strategies = [lump_sum, annuity, phased, hybrid]
        
        # Display comparison table
        comparison_data = []
        for strategy in strategies:
            comparison_data.append({
                'Strategy': strategy['strategy'],
                'Annual Income (₹)': f"₹{strategy['annual_income']:,.0f}",
                'Real Income (Today\'s Value)': f"₹{strategy['real_annual_income']:,.0f}",
                'Total Lifetime Income': f"₹{strategy['total_lifetime_income']:,.0f}",
                'Tax Impact': f"₹{strategy['tax_impact']:,.0f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualize comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Annual Income Comparison
            fig = px.bar(
                x=[s['strategy'] for s in strategies],
                y=[s['annual_income'] for s in strategies],
                title="Annual Income Comparison",
                labels={'x': 'Strategy', 'y': 'Annual Income (₹)'},
                color=[s['annual_income'] for s in strategies],
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Total Lifetime Income Comparison
            fig = px.bar(
                x=[s['strategy'] for s in strategies],
                y=[s['total_lifetime_income'] for s in strategies],
                title="Total Lifetime Income Comparison",
                labels={'x': 'Strategy', 'y': 'Total Lifetime Income (₹)'},
                color=[s['total_lifetime_income'] for s in strategies],
                color_continuous_scale='plasma'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Pros and Cons
        st.subheader("Strategy Details")
        for strategy in strategies:
            with st.expander(f"{strategy['strategy']} - Pros & Cons"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Pros:**")
                    for pro in strategy['pros']:
                        st.write(f"✅ {pro}")
                with col2:
                    st.write("**Cons:**")
                    for con in strategy['cons']:
                        st.write(f"❌ {con}")
    
    with tab2:
        st.header("AI-Powered Recommendation")
        
        # Get AI recommendation
        recommendation = optimizer.get_ai_recommendation(profile, strategies)
        
        # Display recommendation
        st.markdown(f"""
        <div class="recommendation-box">
            <h3>🎯 Recommended Strategy: {recommendation['recommended_strategy']}</h3>
            <h4>Confidence Score: {recommendation['confidence_score']:.0f}%</h4>
            <p><strong>Reasoning:</strong> {recommendation['reasoning']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Strategy scores
        st.subheader("Strategy Scoring")
        scores_df = pd.DataFrame(list(recommendation['scores'].items()), columns=['Strategy', 'AI Score'])
        fig = px.bar(scores_df, x='Strategy', y='AI Score', title="AI Strategy Scores", 
                     color='AI Score', color_continuous_scale='greens')
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommended strategy details
        recommended_strategy_data = next(s for s in strategies if s['strategy'] == recommendation['recommended_strategy'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Annual Income", f"₹{recommended_strategy_data['annual_income']:,.0f}")
        with col2:
            st.metric("Real Income (Today's Value)", f"₹{recommended_strategy_data['real_annual_income']:,.0f}")
        with col3:
            st.metric("Total Lifetime Income", f"₹{recommended_strategy_data['total_lifetime_income']:,.0f}")
        with col4:
            st.metric("Tax Impact", f"₹{recommended_strategy_data['tax_impact']:,.0f}")
    
    with tab3:
        st.header("Income Projections")
        
        years_to_retirement = profile.expected_retirement_age - profile.age
        life_expectancy = optimizer.calculate_life_expectancy(profile.age, 'Male', profile.health_status)
        retirement_years = life_expectancy - profile.expected_retirement_age
        
        # Create projection timeline
        years = list(range(profile.age, life_expectancy + 1))
        
        # Project income for each strategy
        projections = {}
        for strategy in strategies:
            income_stream = []
            for year in years:
                if year < profile.expected_retirement_age:
                    income_stream.append(0)  # No pension income before retirement
                else:
                    # Apply inflation adjustment
                    years_from_retirement = year - profile.expected_retirement_age
                    inflation_factor = (1 + optimizer.inflation_rate) ** years_from_retirement
                    
                    if 'Annuity' in strategy['strategy'] and 'Hybrid' not in strategy['strategy']:
                        # Fixed annuity - loses value to inflation
                        income_stream.append(strategy['annual_income'] / inflation_factor)
                    else:
                        # Variable income - assume some inflation protection
                        income_stream.append(strategy['annual_income'] * (0.8 + 0.2 * inflation_factor))
            
            projections[strategy['strategy']] = income_stream
        
        # Create projection chart
        proj_df = pd.DataFrame({
            'Age': years,
            **projections
        })
        
        fig = px.line(proj_df, x='Age', y=proj_df.columns[1:], 
                      title='Projected Annual Income by Age',
                      labels={'value': 'Annual Income (₹)', 'variable': 'Strategy'})
        fig.add_vline(x=profile.expected_retirement_age, line_dash="dash", 
                      annotation_text="Retirement Age")
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative income chart
        cumulative_projections = {}
        for strategy_name, income_stream in projections.items():
            cumulative = []
            total = 0
            for income in income_stream:
                total += income
                cumulative.append(total)
            cumulative_projections[strategy_name] = cumulative
        
        cumulative_df = pd.DataFrame({
            'Age': years,
            **cumulative_projections
        })
        
        fig2 = px.line(cumulative_df, x='Age', y=cumulative_df.columns[1:], 
                       title='Cumulative Income by Age',
                       labels={'value': 'Cumulative Income (₹)', 'variable': 'Strategy'})
        fig2.add_vline(x=profile.expected_retirement_age, line_dash="dash", 
                       annotation_text="Retirement Age")
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        st.header("What-If Analysis")
        
        st.subheader("Scenario Testing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Market Conditions**")
            market_scenario = st.selectbox("Market Scenario", 
                                         ["Optimistic (+2%)", "Base Case", "Pessimistic (-2%)", "Crisis (-5%)"])
            
            inflation_scenario = st.selectbox("Inflation Scenario",
                                            ["Low (4%)", "Base (6%)", "High (8%)", "Very High (10%)"])
        
        with col2:
            st.write("**Personal Scenarios**")
            retirement_age_adjustment = st.slider("Retirement Age Adjustment", -5, 5, 0)
            health_scenario = st.selectbox("Health Scenario", 
                                         ["Better than expected (+5 years)", "As expected", "Worse than expected (-3 years)"])
        
        # Apply scenarios
        scenario_optimizer = PensionOptimizer()
        
        # Adjust market returns
        if market_scenario == "Optimistic (+2%)":
            scenario_optimizer.market_return += 0.02
            scenario_optimizer.safe_return += 0.01
        elif market_scenario == "Pessimistic (-2%)":
            scenario_optimizer.market_return -= 0.02
            scenario_optimizer.safe_return -= 0.01
        elif market_scenario == "Crisis (-5%)":
            scenario_optimizer.market_return -= 0.05
            scenario_optimizer.safe_return -= 0.02
        
        # Adjust inflation
        if inflation_scenario == "Low (4%)":
            scenario_optimizer.inflation_rate = 0.04
        elif inflation_scenario == "High (8%)":
            scenario_optimizer.inflation_rate = 0.08
        elif inflation_scenario == "Very High (10%)":
            scenario_optimizer.inflation_rate = 0.10
        
        # Create scenario profile
        scenario_profile = profile
        scenario_profile.expected_retirement_age += retirement_age_adjustment
        
        # Recalculate strategies with scenarios
        scenario_lump_sum = scenario_optimizer.simulate_lump_sum_strategy(scenario_profile)
        scenario_annuity = scenario_optimizer.simulate_annuity_strategy(scenario_profile)
        scenario_phased = scenario_optimizer.simulate_phased_withdrawal_strategy(scenario_profile)
        scenario_hybrid = scenario_optimizer.simulate_hybrid_strategy(scenario_profile)
        
        scenario_strategies = [scenario_lump_sum, scenario_annuity, scenario_phased, scenario_hybrid]
        
        # Compare base case vs scenario
        st.subheader("Scenario Impact")
        
        comparison_data = []
        for i, (base, scenario) in enumerate(zip(strategies, scenario_strategies)):
            impact = ((scenario['annual_income'] - base['annual_income']) / base['annual_income']) * 100
            comparison_data.append({
                'Strategy': base['strategy'],
                'Base Case Income': f"₹{base['annual_income']:,.0f}",
                'Scenario Income': f"₹{scenario['annual_income']:,.0f}",
                'Impact (%)': f"{impact:+.1f}%"
            })
        
        scenario_df = pd.DataFrame(comparison_data)
        st.dataframe(scenario_df, use_container_width=True)
        
        # Visualization
        impact_values = []
        strategy_names = []
        for base, scenario in zip(strategies, scenario_strategies):
            impact = ((scenario['annual_income'] - base['annual_income']) / base['annual_income']) * 100
            impact_values.append(impact)
            strategy_names.append(base['strategy'])
        
        fig = px.bar(x=strategy_names, y=impact_values, 
                     title="Scenario Impact on Annual Income (%)",
                     labels={'x': 'Strategy', 'y': 'Impact (%)'},
                     color=impact_values,
                     color_continuous_scale='RdYlGn')
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk analysis
        st.subheader("Risk Analysis")
        st.markdown("""
        <div class="warning-box">
            <h4>Key Risk Factors</h4>
            <ul>
                <li><strong>Market Risk:</strong> Investment returns can be volatile</li>
                <li><strong>Inflation Risk:</strong> Fixed payments lose purchasing power</li>
                <li><strong>Longevity Risk:</strong> Living longer than expected can exhaust savings</li>
                <li><strong>Regulatory Risk:</strong> Tax laws and pension rules may change</li>
                <li><strong>Health Risk:</strong> Medical expenses can impact retirement funds</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tab5:
        st.header("Government Schemes Integration")
        
        st.subheader("Indian Pension Schemes")
        
        # EPF Calculator
        with st.expander("Employee Provident Fund (EPF) - Details"):
            st.write("""
            **Current EPF Rules (2024-25):**
            - Employee contribution: 12% of basic salary
            - Employer contribution: 12% of basic salary (3.67% to EPF, 8.33% to EPS)
            - Current interest rate: ~8.15% per annum
            - Tax-free maturity after 5 years of continuous service
            - Partial withdrawals allowed for specific purposes
            """)
            
            epf_balance = st.number_input("Current EPF Balance (₹)", min_value=0, value=800000)
            monthly_epf = (profile.current_salary * 0.12) / 12  # 12% of salary
            
            epf_maturity = optimizer.project_pension_growth(
                epf_balance, monthly_epf, 
                profile.expected_retirement_age - profile.age, 
                0.0815  # EPF interest rate
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current EPF Balance", f"₹{epf_balance:,.0f}")
            with col2:
                st.metric("Monthly Contribution", f"₹{monthly_epf:,.0f}")
            with col3:
                st.metric("EPF at Retirement", f"₹{epf_maturity:,.0f}")
        
        # NPS Calculator
        with st.expander("National Pension System (NPS) - Details"):
            st.write("""
            **NPS Benefits:**
            - Additional tax deduction up to ₹50,000 under Section 80CCD(1B)
            - Market-linked returns
            - 60% lump sum withdrawal (tax-free), 40% annuity (taxable)
            - Low cost structure
            - Professional fund management
            """)
            
            nps_balance = st.number_input("Current NPS Balance (₹)", min_value=0, value=500000)
            monthly_nps = st.number_input("Monthly NPS Contribution (₹)", min_value=0, value=5000)
            
            nps_maturity = optimizer.project_pension_growth(
                nps_balance, monthly_nps,
                profile.expected_retirement_age - profile.age,
                0.12  # Assumed NPS return
            )
            
            nps_lump_sum = nps_maturity * 0.6  # 60% lump sum
            nps_annuity_corpus = nps_maturity * 0.4  # 40% for annuity
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NPS at Retirement", f"₹{nps_maturity:,.0f}")
            with col2:
                st.metric("Lump Sum (60%)", f"₹{nps_lump_sum:,.0f}")
            with col3:
                st.metric("Annuity Corpus (40%)", f"₹{nps_annuity_corpus:,.0f}")
        
        # Social Security equivalent (Indian context)
        with st.expander("Additional Government Benefits"):
            st.write("""
            **Other Government Schemes:**
            - **Pradhan Mantri Vaya Vandana Yojana (PMVVY):** Guaranteed pension scheme for senior citizens
            - **Atal Pension Yojana (APY):** Guaranteed minimum pension (₹1,000 to ₹5,000)
            - **Senior Citizen Savings Scheme (SCSS):** Higher interest rates for 60+ age group
            - **Tax Benefits:** Section 80C, 80CCD(1), 80CCD(1B), 80CCD(2)
            """)
        
        # Combined government benefits projection
        st.subheader("Total Government Benefits Projection")
        
        total_govt_corpus = epf_maturity + nps_maturity
        govt_annual_income = total_govt_corpus * 0.06  # Conservative 6% withdrawal
        
        # Create comprehensive comparison including government schemes
        comprehensive_data = []
        for strategy in strategies:
            total_income = strategy['annual_income'] + govt_annual_income
            comprehensive_data.append({
                'Strategy': strategy['strategy'],
                'Pension Income': f"₹{strategy['annual_income']:,.0f}",
                'Government Benefits': f"₹{govt_annual_income:,.0f}",
                'Total Annual Income': f"₹{total_income:,.0f}",
                'Replacement Ratio': f"{(total_income/profile.current_salary)*100:.1f}%"
            })
        
        comprehensive_df = pd.DataFrame(comprehensive_data)
        st.dataframe(comprehensive_df, use_container_width=True)
        
        # Replacement ratio visualization
        replacement_ratios = []
        strategy_names = []
        for strategy in strategies:
            total_income = strategy['annual_income'] + govt_annual_income
            replacement_ratio = (total_income / profile.current_salary) * 100
            replacement_ratios.append(replacement_ratio)
            strategy_names.append(strategy['strategy'])
        
        fig = px.bar(x=strategy_names, y=replacement_ratios,
                     title="Income Replacement Ratio (% of Current Salary)",
                     labels={'x': 'Strategy', 'y': 'Replacement Ratio (%)'},
                     color=replacement_ratios,
                     color_continuous_scale='greens')
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                      annotation_text="Recommended 70% replacement")
        st.plotly_chart(fig, use_container_width=True)
        
        # Tax optimization tips
        st.subheader("Tax Optimization Tips")
        st.markdown("""
        <div class="recommendation-box">
            <h4>💡 Tax-Smart Retirement Planning</h4>
            <ul>
                <li><strong>Maximize Section 80C:</strong> Use EPF, PPF, ELSS, and life insurance (up to ₹1.5 lakh)</li>
                <li><strong>NPS Additional Benefit:</strong> Extra ₹50,000 deduction under Section 80CCD(1B)</li>
                <li><strong>Employer NPS:</strong> Section 80CCD(2) allows employer contribution (up to 10% of salary)</li>
                <li><strong>Health Insurance:</strong> Section 80D covers health premiums (up to ₹50,000 for senior citizens)</li>
                <li><strong>Retirement Planning:</strong> Start early to benefit from power of compounding</li>
                <li><strong>Asset Allocation:</strong> Balance between equity and debt based on age and risk appetite</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This tool provides estimates based on current assumptions and should not be considered as financial advice. 
    Please consult with a qualified financial advisor for personalized retirement planning. Market returns, tax rates, and 
    government policies may change over time.
    
    **Data Sources:** EPF interest rates, NPS performance, and tax regulations are based on current Indian financial regulations as of 2024-25.
    """)

# Additional helper functions
def create_sample_data():
    """Create sample user data for demonstration"""
    return {
        "Conservative Investor": {
            "age": 45,
            "current_salary": 1500000,
            "years_of_service": 20,
            "expected_retirement_age": 60,
            "risk_tolerance": "Conservative",
            "current_pension_balance": 3000000,
            "monthly_contributions": 20000,
            "other_savings": 2000000
        },
        "Moderate Investor": {
            "age": 40,
            "current_salary": 1200000,
            "years_of_service": 15,
            "expected_retirement_age": 62,
            "risk_tolerance": "Moderate",
            "current_pension_balance": 2000000,
            "monthly_contributions": 15000,
            "other_savings": 1500000
        },
        "Aggressive Investor": {
            "age": 35,
            "current_salary": 1800000,
            "years_of_service": 10,
            "expected_retirement_age": 58,
            "risk_tolerance": "Aggressive",
            "current_pension_balance": 1500000,
            "monthly_contributions": 25000,
            "other_savings": 3000000
        }
    }

# Quick demo button
if st.sidebar.button("📊 Load Sample Data"):
    sample_data = create_sample_data()
    selected_sample = st.sidebar.selectbox("Choose Sample Profile", list(sample_data.keys()))
    st.sidebar.write(f"Loaded profile: {selected_sample}")
    st.sidebar.json(sample_data[selected_sample])

# Export functionality
if st.sidebar.button("📁 Export Analysis"):
    st.sidebar.success("Analysis exported! (Demo - in real app, this would download a PDF report)")

if __name__ == "__main__":
    main()