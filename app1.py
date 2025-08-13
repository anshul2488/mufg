import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import datetime
from dataclasses import dataclass
from typing import Dict, List
import hashlib
import time
from cryptography.fernet import Fernet
import io

st.set_page_config(page_title="PensionWise.AI", page_icon="💰", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem;font-weight: bold;color: #1f77b4;text-align: center;margin-bottom: 2rem;}
    .metric-container {background-color: #f0f2f6;padding: 1rem;border-radius: 0.5rem;margin: 0.5rem 0;}
    .recommendation-box {background-color: #e8f5e8;padding: 1.5rem;border-radius: 0.5rem;border-left: 5px solid #28a745;margin: 1rem 0;}
    .warning-box {background-color: #fff3cd;padding: 1rem;border-radius: 0.5rem;border-left: 5px solid #ffc107;margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

USER_CREDENTIALS = {
    "admin": hashlib.sha256("adminpass".encode()).hexdigest(),
    "user": hashlib.sha256("userpass".encode()).hexdigest()
}

SESSION_TIMEOUT = 900

if "fernet_key" not in st.session_state:
    st.session_state.fernet_key = Fernet.generate_key().decode()
fernet_key = st.session_state.fernet_key
cipher = Fernet(fernet_key.encode())

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

def login():
    st.sidebar.subheader("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        hashed_pass = hashlib.sha256(password.encode()).hexdigest()
        if USER_CREDENTIALS.get(username) == hashed_pass:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["last_active"] = time.time()
            st.sidebar.success(f"Welcome {username}!")
        else:
            st.sidebar.error("Invalid username or password")

if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    login()
    st.stop()

if time.time() - st.session_state.get("last_active", time.time()) > SESSION_TIMEOUT:
    st.session_state.authenticated = False
    st.warning("Session expired. Please log in again.")
    st.stop()

st.session_state.last_active = time.time()

def mask_currency(amount):
    try:
        amount = float(amount)
    except Exception:
        return "₹--"
    return f"₹{amount:,.0f}" if amount < 10000000 else "₹XX,XX,XXX"

def encrypt_bytes(data_bytes: bytes) -> bytes:
    return cipher.encrypt(data_bytes)

def validate_inputs(profile):
    errors = []
    if profile.age >= profile.expected_retirement_age:
        errors.append("Retirement age must be greater than current age.")
    if profile.current_salary <= 0:
        errors.append("Salary must be greater than zero.")
    if profile.monthly_contributions < 0:
        errors.append("Monthly contributions cannot be negative.")
    if profile.current_pension_balance < 0:
        errors.append("Current pension balance cannot be negative.")
    return errors

class PensionOptimizer:
    def __init__(self):
        self.inflation_rate = 0.06
        self.market_return = 0.10
        self.safe_return = 0.07
        self.tax_rates = {'low': 0.10, 'medium': 0.20, 'high': 0.30}
    def calculate_life_expectancy(self, age: int, gender: str, health: str) -> int:
        base_expectancy = 78 if gender == 'Male' else 82
        health_adjustment = {'Excellent': 5, 'Good': 2, 'Fair': 0, 'Poor': -3}
        return base_expectancy + health_adjustment.get(health, 0)
    def project_pension_growth(self, current_balance: float, monthly_contrib: float, years_to_retirement: int, return_rate: float) -> float:
        months = max(0, years_to_retirement) * 12
        monthly_rate = return_rate / 12
        fv_current = current_balance * (1 + return_rate) ** max(0, years_to_retirement)
        if monthly_rate > 0 and months > 0:
            fv_contributions = monthly_contrib * (((1 + monthly_rate) ** months - 1) / monthly_rate)
        else:
            fv_contributions = monthly_contrib * months
        return max(0.0, fv_current + fv_contributions)
    def calculate_annuity_payment(self, principal: float, years: int, return_rate: float) -> float:
        if years <= 0:
            return 0.0
        if return_rate == 0:
            return principal / years
        return principal * (return_rate * (1 + return_rate) ** years) / ((1 + return_rate) ** years - 1)
    def simulate_lump_sum_strategy(self, profile: UserProfile) -> Dict:
        years_to_retirement = max(0, profile.expected_retirement_age - profile.age)
        life_expectancy = self.calculate_life_expectancy(profile.age, 'Male', profile.health_status)
        retirement_years = max(1, life_expectancy - profile.expected_retirement_age)
        pension_at_retirement = self.project_pension_growth(profile.current_pension_balance, profile.monthly_contributions, years_to_retirement, self.market_return)
        tax_rate = self.tax_rates['high']
        after_tax_amount = pension_at_retirement * (1 - tax_rate)
        annual_withdrawal = self.calculate_annuity_payment(after_tax_amount, retirement_years, self.safe_return)
        real_annual_income = annual_withdrawal / ((1 + self.inflation_rate) ** years_to_retirement) if years_to_retirement > 0 else annual_withdrawal
        return {'strategy': 'Lump Sum','pension_at_retirement': pension_at_retirement,'after_tax_amount': after_tax_amount,'annual_income': annual_withdrawal,'real_annual_income': real_annual_income,'total_lifetime_income': annual_withdrawal * retirement_years,'tax_impact': pension_at_retirement - after_tax_amount,'pros': ['Full control over investments','Potential for higher returns','Inheritance benefit'],'cons': ['Market risk','Inflation risk','High upfront tax burden','Longevity risk']}
    def simulate_annuity_strategy(self, profile: UserProfile) -> Dict:
        years_to_retirement = max(0, profile.expected_retirement_age - profile.age)
        life_expectancy = self.calculate_life_expectancy(profile.age, 'Male', profile.health_status)
        retirement_years = max(1, life_expectancy - profile.expected_retirement_age)
        pension_at_retirement = self.project_pension_growth(profile.current_pension_balance, profile.monthly_contributions, years_to_retirement, self.market_return)
        tax_rate = self.tax_rates['medium']
        annual_gross_payment = self.calculate_annuity_payment(pension_at_retirement, retirement_years, 0.04)
        annual_after_tax = annual_gross_payment * (1 - tax_rate)
        real_annual_income = annual_after_tax / ((1 + self.inflation_rate) ** years_to_retirement) if years_to_retirement > 0 else annual_after_tax
        return {'strategy': 'Annuity','pension_at_retirement': pension_at_retirement,'annual_gross_payment': annual_gross_payment,'annual_income': annual_after_tax,'real_annual_income': real_annual_income,'total_lifetime_income': annual_after_tax * retirement_years,'tax_impact': annual_gross_payment * tax_rate * retirement_years,'pros': ['Guaranteed income for life','Protection from market volatility','Lower tax rate'],'cons': ['No inflation protection','No inheritance benefit','Lower potential returns']}
    def simulate_phased_withdrawal_strategy(self, profile: UserProfile) -> Dict:
        years_to_retirement = max(0, profile.expected_retirement_age - profile.age)
        life_expectancy = self.calculate_life_expectancy(profile.age, 'Male', profile.health_status)
        retirement_years = max(1, life_expectancy - profile.expected_retirement_age)
        pension_at_retirement = self.project_pension_growth(profile.current_pension_balance, profile.monthly_contributions, years_to_retirement, self.market_return)
        annual_withdrawal_rate = 0.045
        annual_gross_withdrawal = pension_at_retirement * annual_withdrawal_rate
        tax_rate = self.tax_rates['low']
        annual_after_tax = annual_gross_withdrawal * (1 - tax_rate)
        real_annual_income = annual_after_tax / ((1 + self.inflation_rate) ** years_to_retirement) if years_to_retirement > 0 else annual_after_tax
        total_lifetime = annual_after_tax * retirement_years * 1.1
        return {'strategy': 'Phased Withdrawal','pension_at_retirement': pension_at_retirement,'annual_withdrawal_rate': annual_withdrawal_rate,'annual_income': annual_after_tax,'real_annual_income': real_annual_income,'total_lifetime_income': total_lifetime,'tax_impact': annual_gross_withdrawal * tax_rate * retirement_years,'pros': ['Flexible withdrawals','Continued growth potential','Lower tax rates','Inheritance benefit'],'cons': ['Market risk','Requires active management','Longevity risk']}
    def simulate_hybrid_strategy(self, profile: UserProfile) -> Dict:
        years_to_retirement = max(0, profile.expected_retirement_age - profile.age)
        life_expectancy = self.calculate_life_expectancy(profile.age, 'Male', profile.health_status)
        retirement_years = max(1, life_expectancy - profile.expected_retirement_age)
        pension_at_retirement = self.project_pension_growth(profile.current_pension_balance, profile.monthly_contributions, years_to_retirement, self.market_return)
        annuity_portion = pension_at_retirement * 0.6
        investment_portion = pension_at_retirement * 0.4
        annuity_annual = self.calculate_annuity_payment(annuity_portion, retirement_years, 0.04)
        annuity_after_tax = annuity_annual * (1 - self.tax_rates['medium'])
        investment_annual = investment_portion * 0.045
        investment_after_tax = investment_annual * (1 - self.tax_rates['low'])
        total_annual = annuity_after_tax + investment_after_tax
        real_annual_income = total_annual / ((1 + self.inflation_rate) ** years_to_retirement) if years_to_retirement > 0 else total_annual
        return {'strategy': 'Hybrid (60% Annuity + 40% Investment)','pension_at_retirement': pension_at_retirement,'annuity_portion': annuity_portion,'investment_portion': investment_portion,'annual_income': total_annual,'real_annual_income': real_annual_income,'total_lifetime_income': total_annual * retirement_years * 1.05,'tax_impact': (annuity_annual * self.tax_rates['medium'] + investment_annual * self.tax_rates['low']) * retirement_years,'pros': ['Balanced risk','Some guaranteed income','Growth potential','Partial inheritance'],'cons': ['Complexity','Moderate market risk','Requires monitoring']}
    def get_ai_recommendation(self, profile: UserProfile, strategies: List[Dict]) -> Dict:
        scores = {}
        for strategy in strategies:
            score = 0
            strategy_name = strategy['strategy']
            score += (strategy.get('total_lifetime_income', 0) / 1000000)
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
            else:
                if 'Hybrid' in strategy_name:
                    score += 15
                elif 'Phased' in strategy_name:
                    score += 10
            if profile.health_status in ['Excellent', 'Good']:
                if 'Lump Sum' in strategy_name or 'Phased' in strategy_name:
                    score += 10
            else:
                if 'Annuity' in strategy_name:
                    score += 15
            if profile.dependents > 0:
                if 'Lump Sum' in strategy_name or 'Phased' in strategy_name:
                    score += 5
            if profile.other_savings > profile.current_salary * 5:
                if 'Annuity' in strategy_name:
                    score += 10
            else:
                if 'Phased' in strategy_name or 'Hybrid' in strategy_name:
                    score += 5
            scores[strategy_name] = score
        best_strategy = max(scores, key=scores.get)
        confidence = min(95, max(60, scores[best_strategy] * 2))
        return {'recommended_strategy': best_strategy, 'confidence_score': confidence, 'reasoning': self.get_recommendation_reasoning(profile, best_strategy), 'scores': scores}
    def get_recommendation_reasoning(self, profile: UserProfile, strategy: str) -> str:
        reasons = []
        if 'Annuity' in strategy:
            reasons.append("Annuity provides guaranteed income security")
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

def create_sample_data():
    return {
        "Conservative Investor": {"age": 45,"current_salary": 1500000,"years_of_service": 20,"expected_retirement_age": 60,"risk_tolerance": "Conservative","current_pension_balance": 3000000,"monthly_contributions": 20000,"other_savings": 2000000},
        "Moderate Investor": {"age": 40,"current_salary": 1200000,"years_of_service": 15,"expected_retirement_age": 62,"risk_tolerance": "Moderate","current_pension_balance": 2000000,"monthly_contributions": 15000,"other_savings": 1500000},
        "Aggressive Investor": {"age": 35,"current_salary": 1800000,"years_of_service": 10,"expected_retirement_age": 58,"risk_tolerance": "Aggressive","current_pension_balance": 1500000,"monthly_contributions": 25000,"other_savings": 3000000}
    }

def export_encrypted_report(profile: UserProfile, strategies: List[Dict], recommendation: Dict):
    out = io.StringIO()
    out.write("Profile\n")
    for k, v in profile.__dict__.items():
        out.write(f"{k},{v}\n")
    out.write("\nStrategies\n")
    for s in strategies:
        out.write(f"Strategy,{s['strategy']}\n")
        for k, v in s.items():
            if k == 'strategy':
                continue
            out.write(f"{k},{v}\n")
        out.write("\n")
    out.write("Recommendation\n")
    for k, v in recommendation.items():
        out.write(f"{k},{v}\n")
    raw_bytes = out.getvalue().encode()
    encrypted = encrypt_bytes(raw_bytes)
    return encrypted

def export_masked_csv(profile: UserProfile, strategies: List[Dict]):
    rows = []
    for s in strategies:
        rows.append({
            "Strategy": s["strategy"],
            "Annual Income": mask_currency(s.get("annual_income", 0)),
            "Real Annual Income": mask_currency(s.get("real_annual_income", 0)),
            "Total Lifetime Income": mask_currency(s.get("total_lifetime_income", 0)),
            "Tax Impact": mask_currency(s.get("tax_impact", 0))
        })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode()

def try_load_llama(model_path: str):
    try:
        from llama_cpp import Llama
        return Llama(model_path=model_path)
    except Exception as e:
        return None

def get_llm_response(model_handle, prompt: str, max_tokens: int = 256):
    if model_handle is None:
        return fallback_llm_response(prompt)
    try:
        resp = model_handle.create(prompt=prompt, max_tokens=max_tokens, stop=["\n\n"])
        text = resp.get("choices", [{}])[0].get("text", "")
        return text.strip() if text else fallback_llm_response(prompt)
    except Exception:
        return fallback_llm_response(prompt)

def fallback_llm_response(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if "recommend" in prompt_lower or "what should i do" in prompt_lower:
        return "Based on your profile, the optimizer compares Lump Sum, Annuity, Phased Withdrawal, and Hybrid strategies. It scores them using projected total lifetime income, risk tolerance, health, dependents, and other savings. Use the AI Recommendation tab to see the best match and reasoning."
    if "explain" in prompt_lower or "why" in prompt_lower:
        return "The recommendation logic weighs guaranteed income (annuity) versus growth potential (lump sum), adjusting for risk tolerance and health. Dependents and other savings shift preference toward liquidity or inheritance."
    return "I can explain results, convert numbers to plain language, or walk through calculations step-by-step. Try asking: 'Explain why hybrid is recommended' or 'Show calculation steps for lump sum projection'."

def main():
    st.markdown('<h1 class="main-header">🎯 AI Pension Benefits Optimizer</h1>', unsafe_allow_html=True)
    optimizer = PensionOptimizer()
    with st.sidebar:
        st.header("👤 Your Profile")
        st.subheader("Personal Details")
        age = st.slider("Current Age", 25, 65, 45)
        expected_retirement_age = st.slider("Expected Retirement Age", age + 1, 75, 60)
        current_salary = st.number_input("Annual Salary (₹)", min_value=100000, max_value=10000000, value=1200000, step=50000)
        years_of_service = st.slider("Years of Service", 0, 40, 15)
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        dependents = st.slider("Number of Dependents", 0, 5, 2)
        location = st.selectbox("Location", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Other"])
        st.subheader("Financial Details")
        current_pension_balance = st.number_input("Current Pension Balance (₹)", min_value=0, max_value=50000000, value=2500000, step=100000)
        monthly_contributions = st.number_input("Monthly Contributions (₹)", min_value=0, max_value=100000, value=15000, step=1000)
        other_savings = st.number_input("Other Savings/Investments (₹)", min_value=0, max_value=50000000, value=1500000, step=100000)
        spouse_pension = st.number_input("Spouse's Pension Value (₹)", min_value=0, max_value=10000000, value=500000, step=50000)
        st.subheader("Preferences & Health")
        pension_plan_type = st.selectbox("Pension Plan Type", ["Employee Provident Fund (EPF)", "National Pension System (NPS)", "Corporate Pension", "Government Pension"])
        risk_tolerance = st.selectbox("Risk Tolerance", ["Conservative", "Moderate", "Aggressive"])
        health_status = st.selectbox("Health Status", ["Excellent", "Good", "Fair", "Poor"])
        st.markdown("---")
        if st.button("📊 Load Sample Data"):
            sample_data = create_sample_data()
            selected_sample = st.selectbox("Choose Sample Profile", list(sample_data.keys()))
            if st.button("Load Selected Sample"):
                sd = sample_data[selected_sample]
                st.session_state.sample = sd
                st.experimental_rerun()
        st.markdown("---")
        if st.button("🔒 Logout"):
            st.session_state.authenticated = False
            st.rerun()
        st.markdown("---")
        st.subheader("LLM Assistant")
        enable_llm = st.checkbox("Enable local LLaMA assistant (optional)", value=False)
        model_path = st.text_input("Local model path (e.g. /path/to/ggml-model-q4_0.bin)", value="")
        if enable_llm and model_path:
            if "llama_handle" not in st.session_state or st.session_state.get("llama_model_path") != model_path:
                llama_handle = try_load_llama(model_path)
                st.session_state.llama_handle = llama_handle
                st.session_state.llama_model_path = model_path
            else:
                llama_handle = st.session_state.get("llama_handle")
        else:
            llama_handle = None
    if "sample" in st.session_state:
        sd = st.session_state.sample
        age = sd.get("age", age)
        expected_retirement_age = sd.get("expected_retirement_age", expected_retirement_age)
        current_salary = sd.get("current_salary", current_salary)
        years_of_service = sd.get("years_of_service", years_of_service)
        current_pension_balance = sd.get("current_pension_balance", current_pension_balance)
        monthly_contributions = sd.get("monthly_contributions", monthly_contributions)
        other_savings = sd.get("other_savings", other_savings)
    profile = UserProfile(age=age, current_salary=current_salary, years_of_service=years_of_service, expected_retirement_age=expected_retirement_age, marital_status=marital_status, dependents=dependents, location=location, pension_plan_type=pension_plan_type, current_pension_balance=current_pension_balance, monthly_contributions=monthly_contributions, risk_tolerance=risk_tolerance, health_status=health_status, other_savings=other_savings, spouse_pension=spouse_pension)
    errors = validate_inputs(profile)
    if errors:
        for err in errors:
            st.error(err)
        st.stop()
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Strategy Comparison", "🎯 AI Recommendation", "📈 Projections", "🔍 What-If Analysis", "📋 Government Schemes", "🤖 LLM Assistant"])
    with st.spinner("Analyzing pension strategies..."):
        lump_sum = optimizer.simulate_lump_sum_strategy(profile)
        annuity = optimizer.simulate_annuity_strategy(profile)
        phased = optimizer.simulate_phased_withdrawal_strategy(profile)
        hybrid = optimizer.simulate_hybrid_strategy(profile)
        strategies = [lump_sum, annuity, phased, hybrid]
    with tab1:
        st.header("Strategy Comparison")
        comparison_data = []
        for strategy in strategies:
            comparison_data.append({
                'Strategy': strategy['strategy'],
                'Annual Income (₹)': mask_currency(strategy['annual_income']),
                "Real Income (Today's Value)": mask_currency(strategy['real_annual_income']),
                'Total Lifetime Income': mask_currency(strategy['total_lifetime_income']),
                'Tax Impact': mask_currency(strategy['tax_impact'])
            })
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(x=[s['strategy'] for s in strategies], y=[s['annual_income'] for s in strategies], title="Annual Income Comparison", labels={'x': 'Strategy', 'y': 'Annual Income (₹)'}, color=[s['annual_income'] for s in strategies], color_continuous_scale='viridis')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(x=[s['strategy'] for s in strategies], y=[s['total_lifetime_income'] for s in strategies], title="Total Lifetime Income Comparison", labels={'x': 'Strategy', 'y': 'Total Lifetime Income (₹)'}, color=[s['total_lifetime_income'] for s in strategies], color_continuous_scale='plasma')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
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
        recommendation = optimizer.get_ai_recommendation(profile, strategies)
        st.markdown(f"""<div class="recommendation-box"><h3>🎯 Recommended Strategy: {recommendation['recommended_strategy']}</h3><h4>Confidence Score: {recommendation['confidence_score']:.0f}%</h4><p><strong>Reasoning:</strong> {recommendation['reasoning']}</p></div>""", unsafe_allow_html=True)
        st.subheader("Strategy Scoring")
        scores_df = pd.DataFrame(list(recommendation['scores'].items()), columns=['Strategy', 'AI Score'])
        fig = px.bar(scores_df, x='Strategy', y='AI Score', title="AI Strategy Scores", color='AI Score', color_continuous_scale='greens')
        st.plotly_chart(fig, use_container_width=True)
        recommended_strategy_data = next(s for s in strategies if s['strategy'] == recommendation['recommended_strategy'])
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Annual Income", mask_currency(recommended_strategy_data['annual_income']))
        with col2:
            st.metric("Real Income (Today's Value)", mask_currency(recommended_strategy_data['real_annual_income']))
        with col3:
            st.metric("Total Lifetime Income", mask_currency(recommended_strategy_data['total_lifetime_income']))
        with col4:
            st.metric("Tax Impact", mask_currency(recommended_strategy_data['tax_impact']))
    with tab3:
        st.header("Income Projections")
        years_to_retirement = max(0, profile.expected_retirement_age - profile.age)
        life_expectancy = optimizer.calculate_life_expectancy(profile.age, 'Male', profile.health_status)
        years = list(range(profile.age, life_expectancy + 1))
        projections = {}
        for strategy in strategies:
            income_stream = []
            for year in years:
                if year < profile.expected_retirement_age:
                    income_stream.append(0)
                else:
                    years_from_retirement = year - profile.expected_retirement_age
                    inflation_factor = (1 + optimizer.inflation_rate) ** years_from_retirement
                    if 'Annuity' in strategy['strategy'] and 'Hybrid' not in strategy['strategy']:
                        income_stream.append(strategy['annual_income'] / inflation_factor if inflation_factor > 0 else strategy['annual_income'])
                    else:
                        income_stream.append(strategy['annual_income'] * (0.8 + 0.2 * inflation_factor))
            projections[strategy['strategy']] = income_stream
        proj_df = pd.DataFrame({'Age': years, **projections})
        fig = px.line(proj_df, x='Age', y=proj_df.columns[1:], title='Projected Annual Income by Age', labels={'value': 'Annual Income (₹)', 'variable': 'Strategy'})
        fig.add_vline(x=profile.expected_retirement_age, line_dash="dash", annotation_text="Retirement Age")
        st.plotly_chart(fig, use_container_width=True)
        cumulative_projections = {}
        for strategy_name, income_stream in projections.items():
            cumulative = []
            total = 0
            for income in income_stream:
                total += income
                cumulative.append(total)
            cumulative_projections[strategy_name] = cumulative
        cumulative_df = pd.DataFrame({'Age': years, **cumulative_projections})
        fig2 = px.line(cumulative_df, x='Age', y=cumulative_df.columns[1:], title='Cumulative Income by Age', labels={'value': 'Cumulative Income (₹)', 'variable': 'Strategy'})
        fig2.add_vline(x=profile.expected_retirement_age, line_dash="dash", annotation_text="Retirement Age")
        st.plotly_chart(fig2, use_container_width=True)
    with tab4:
        st.header("What-If Analysis")
        col1, col2 = st.columns(2)
        with col1:
            market_scenario = st.selectbox("Market Scenario", ["Optimistic (+2%)", "Base Case", "Pessimistic (-2%)", "Crisis (-5%)"])
            inflation_scenario = st.selectbox("Inflation Scenario", ["Low (4%)", "Base (6%)", "High (8%)", "Very High (10%)"])
        with col2:
            retirement_age_adjustment = st.slider("Retirement Age Adjustment", -5, 5, 0)
            health_scenario = st.selectbox("Health Scenario", ["Better than expected (+5 years)", "As expected", "Worse than expected (-3 years)"])
        scenario_optimizer = PensionOptimizer()
        if market_scenario == "Optimistic (+2%)":
            scenario_optimizer.market_return += 0.02
            scenario_optimizer.safe_return += 0.01
        elif market_scenario == "Pessimistic (-2%)":
            scenario_optimizer.market_return -= 0.02
            scenario_optimizer.safe_return -= 0.01
        elif market_scenario == "Crisis (-5%)":
            scenario_optimizer.market_return -= 0.05
            scenario_optimizer.safe_return -= 0.02
        if inflation_scenario == "Low (4%)":
            scenario_optimizer.inflation_rate = 0.04
        elif inflation_scenario == "High (8%)":
            scenario_optimizer.inflation_rate = 0.08
        elif inflation_scenario == "Very High (10%)":
            scenario_optimizer.inflation_rate = 0.10
        scenario_profile = UserProfile(**{**profile.__dict__})
        scenario_profile.expected_retirement_age = max(profile.age + 1, profile.expected_retirement_age + retirement_age_adjustment)
        scenario_lump_sum = scenario_optimizer.simulate_lump_sum_strategy(scenario_profile)
        scenario_annuity = scenario_optimizer.simulate_annuity_strategy(scenario_profile)
        scenario_phased = scenario_optimizer.simulate_phased_withdrawal_strategy(scenario_profile)
        scenario_hybrid = scenario_optimizer.simulate_hybrid_strategy(scenario_profile)
        scenario_strategies = [scenario_lump_sum, scenario_annuity, scenario_phased, scenario_hybrid]
        st.subheader("Scenario Impact")
        comparison_data = []
        for base, scenario in zip(strategies, scenario_strategies):
            base_income = base.get('annual_income', 1)
            scenario_income = scenario.get('annual_income', 0)
            impact = ((scenario_income - base_income) / base_income) * 100 if base_income != 0 else 0
            comparison_data.append({'Strategy': base['strategy'], 'Base Case Income': mask_currency(base_income), 'Scenario Income': mask_currency(scenario_income), 'Impact (%)': f"{impact:+.1f}%"})
        scenario_df = pd.DataFrame(comparison_data)
        st.dataframe(scenario_df, use_container_width=True)
        impact_values = []
        strategy_names = []
        for base, scenario in zip(strategies, scenario_strategies):
            base_income = base.get('annual_income', 1)
            scenario_income = scenario.get('annual_income', 0)
            impact = ((scenario_income - base_income) / base_income) * 100 if base_income != 0 else 0
            impact_values.append(impact)
            strategy_names.append(base['strategy'])
        fig = px.bar(x=strategy_names, y=impact_values, title="Scenario Impact on Annual Income (%)", labels={'x': 'Strategy', 'y': 'Impact (%)'}, color=impact_values, color_continuous_scale='RdYlGn')
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="warning-box"><h4>Key Risk Factors</h4><ul><li><strong>Market Risk:</strong> Investment returns can be volatile</li><li><strong>Inflation Risk:</strong> Fixed payments lose purchasing power</li><li><strong>Longevity Risk:</strong> Living longer than expected can exhaust savings</li><li><strong>Regulatory Risk:</strong> Tax laws and pension rules may change</li><li><strong>Health Risk:</strong> Medical expenses can impact retirement funds</li></ul></div>""", unsafe_allow_html=True)
    with tab5:
        st.header("Government Schemes Integration")
        with st.expander("Employee Provident Fund (EPF) - Details"):
            epf_balance = st.number_input("Current EPF Balance (₹)", min_value=0, value=800000)
            monthly_epf = (profile.current_salary * 0.12) / 12
            epf_maturity = optimizer.project_pension_growth(epf_balance, monthly_epf, profile.expected_retirement_age - profile.age, 0.0815)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current EPF Balance", mask_currency(epf_balance))
            with col2:
                st.metric("Monthly Contribution", mask_currency(monthly_epf))
            with col3:
                st.metric("EPF at Retirement", mask_currency(epf_maturity))
        with st.expander("National Pension System (NPS) - Details"):
            nps_balance = st.number_input("Current NPS Balance (₹)", min_value=0, value=500000)
            monthly_nps = st.number_input("Monthly NPS Contribution (₹)", min_value=0, value=5000)
            nps_maturity = optimizer.project_pension_growth(nps_balance, monthly_nps, profile.expected_retirement_age - profile.age, 0.12)
            nps_lump_sum = nps_maturity * 0.6
            nps_annuity_corpus = nps_maturity * 0.4
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("NPS at Retirement", mask_currency(nps_maturity))
            with col2:
                st.metric("Lump Sum (60%)", mask_currency(nps_lump_sum))
            with col3:
                st.metric("Annuity Corpus (40%)", mask_currency(nps_annuity_corpus))
        total_govt_corpus = epf_maturity + nps_maturity
        govt_annual_income = total_govt_corpus * 0.06
        comprehensive_data = []
        for strategy in strategies:
            total_income = strategy['annual_income'] + govt_annual_income
            comprehensive_data.append({'Strategy': strategy['strategy'], 'Pension Income': mask_currency(strategy['annual_income']), 'Government Benefits': mask_currency(govt_annual_income), 'Total Annual Income': mask_currency(total_income), 'Replacement Ratio': f"{(total_income/profile.current_salary)*100:.1f}%"})
        comprehensive_df = pd.DataFrame(comprehensive_data)
        st.dataframe(comprehensive_df, use_container_width=True)
        replacement_ratios = []
        strategy_names = []
        for strategy in strategies:
            total_income = strategy['annual_income'] + govt_annual_income
            replacement_ratio = (total_income / profile.current_salary) * 100
            replacement_ratios.append(replacement_ratio)
            strategy_names.append(strategy['strategy'])
        fig = px.bar(x=strategy_names, y=replacement_ratios, title="Income Replacement Ratio (% of Current Salary)", labels={'x': 'Strategy', 'y': 'Replacement Ratio (%)'}, color=replacement_ratios, color_continuous_scale='greens')
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Recommended 70% replacement")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""<div class="recommendation-box"><h4>💡 Tax-Smart Retirement Planning</h4><ul><li><strong>Maximize Section 80C:</strong> Use EPF, PPF, ELSS, and life insurance (up to ₹1.5 lakh)</li><li><strong>NPS Additional Benefit:</strong> Extra ₹50,000 deduction under Section 80CCD(1B)</li><li><strong>Employer NPS:</strong> Section 80CCD(2) allows employer contribution (up to 10% of salary)</li><li><strong>Health Insurance:</strong> Section 80D covers health premiums (up to ₹50,000 for senior citizens)</li><li><strong>Retirement Planning:</strong> Start early to benefit from power of compounding</li><li><strong>Asset Allocation:</strong> Balance between equity and debt based on age and risk appetite</li></ul></div>""", unsafe_allow_html=True)
    with tab6:
        st.header("LLM Assistant")
        st.write("Use the assistant to ask questions about the calculations, request explanations, or get textual summaries. If you enabled local LLaMA in the sidebar and provided a model path, the assistant will use it.")
        user_prompt = st.text_area("Ask the assistant (e.g. 'Explain why annuity is recommended')", height=120)
        if st.button("Ask Assistant"):
            if not user_prompt or user_prompt.strip() == "":
                st.warning("Please enter a prompt.")
            else:
                model_handle = st.session_state.get("llama_handle") if st.session_state.get("llama_handle") else None
                enriched_prompt = f"""You are a pension assistant. The user's profile: age={profile.age}, retirement_age={profile.expected_retirement_age}, current_pension_balance={profile.current_pension_balance}, monthly_contributions={profile.monthly_contributions}, current_salary={profile.current_salary}, risk_tolerance={profile.risk_tolerance}, health_status={profile.health_status}, dependents={profile.dependents}. Available strategies summary: {', '.join([s['strategy'] for s in strategies])}. User question: {user_prompt}"""
                answer = get_llm_response(model_handle, enriched_prompt, max_tokens=300)
                st.markdown("**Assistant response:**")
                st.write(answer)
    st.markdown("---")
    st.markdown("**Disclaimer:** This tool provides estimates based on current assumptions and should not be considered as financial advice. Consult a qualified advisor for personalized planning.")
    with st.sidebar:
        st.subheader("Export / Download")
        if st.button("📁 Export Encrypted Analysis"):
            encrypted = export_encrypted_report(profile, strategies, recommendation)
            st.download_button(label="Download Encrypted Report", data=encrypted, file_name="analysis_encrypted.bin", mime="application/octet-stream")
            st.success("Encrypted report ready for download")
        if st.button("📄 Export Masked CSV"):
            csv_bytes = export_masked_csv(profile, strategies)
            st.download_button(label="Download Masked CSV", data=csv_bytes, file_name="analysis_masked.csv", mime="text/csv")
            st.success("Masked CSV ready for download")

if __name__ == "__main__":
    main()

