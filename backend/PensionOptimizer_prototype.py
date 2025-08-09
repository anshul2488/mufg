"""
PensionOptimizer_prototype.py

Single-file prototype backend for the Pension Benefits Optimizer (hackathon MVP).
- FastAPI app with a /simulate endpoint
- Monte Carlo simulation for 3 strategies: lump-sum invested, immediate annuity, hybrid
- Simple tax & inflation modeling (configurable)
- Sample users dataset included

Run: pip install fastapi uvicorn numpy pandas scipy pydantic
Then: uvicorn PensionOptimizer_prototype:app --reload

Endpoints:
- POST /simulate  -> JSON input: user profile and simulation settings
  returns: per-strategy metrics and summary timeseries (median & percentiles)

This file is intentionally compact and commented for hackathon speed.
"""
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import json

app = FastAPI(title="Pension Benefits Optimizer - Prototype")

# ---------------------- Data models ----------------------
class UserProfile(BaseModel):
    age: int = Field(..., example=60)
    gender: Optional[str] = Field('M')
    current_salary: Optional[float] = Field(None, example=120000)
    years_service: Optional[int] = Field(None)
    pension_type: Optional[str] = Field('DefinedContribution')
    accrued_pot: float = Field(..., example=300000)
    expected_retirement_age: int = Field(..., example=62)
    marital_status: Optional[str] = Field('Single')
    risk_pref: Optional[str] = Field('Moderate')
    desired_income_pct: Optional[float] = Field(70.0)
    country: Optional[str] = Field('India')
    spouse_age: Optional[int] = None

class SimSettings(BaseModel):
    simulations: int = Field(2000, description='Monte Carlo sims')
    mu: float = Field(0.05, description='Expected annual return (nominal)')
    sigma: float = Field(0.12, description='Annual volatility')
    inflation: float = Field(0.03)
    max_age: int = Field(100)

class SimRequest(BaseModel):
    user: UserProfile
    settings: Optional[SimSettings] = SimSettings()

# ---------------------- Utilities ----------------------
# Simple Gompertz-like mortality sampling (approximation)
def sample_age_of_death(current_age, n, max_age=100):
    # We construct a skewed distribution: survival to older ages declines
    # This is not a substitute for real life tables; replace in production.
    ages = np.arange(current_age, max_age + 1)
    # baseline hazard increasing exponentially
    k = 0.09
    hazard = np.exp(k * (ages - current_age))
    survival = np.exp(-np.cumsum(hazard) / 1000.0)
    survival = survival / survival[0]
    pmf = -np.diff(np.concatenate(([1.0], survival)))
    pmf = pmf / pmf.sum()
    death_ages = np.random.choice(ages, size=n, p=pmf)
    return death_ages

# Simple annuity pricing: given pot and annuity_rate (realistic factor), return annual payment
# annuity_rate: fraction of pot paid annually for life (actuarial), e.g., 0.05
# In practice annuity_rate depends on age, interest rates, guaranteed periods, joint-life etc.

def price_annuity(pot, annuity_rate=0.05):
    return pot * annuity_rate

# Tax function: simple progressive brackets example (replace per-country rules)
def simple_tax(income, country='India'):
    # returns tax on annual income
    if country.lower() == 'india':
        # very simplified brackets
        if income <= 250000:
            return 0.0
        elif income <= 500000:
            return 0.05 * (income - 250000)
        elif income <= 1000000:
            return 12500 + 0.20 * (income - 500000)
        else:
            return 112500 + 0.30 * (income - 1000000)
    else:
        # fallback flat 15% tax
        return 0.15 * income

# Simulate invested lump-sum withdrawals using a simple fixed-percentage withdrawal strategy
# (e.g., 4% initial rule with inflation-adjusted withdrawals)
def simulate_withdrawals(pot, years, mu, sigma, inflation, withdrawal_rate=0.04, sims=2000):
    n_years = len(years)
    results = np.zeros((sims, n_years))
    for s in range(sims):
        balances = np.zeros(n_years + 1)
        balances[0] = pot
        withdraw = withdrawal_rate * pot
        for t in range(n_years):
            # invest growth
            ret = np.random.normal(mu, sigma)
            balances[t+1] = balances[t] * (1 + ret) - withdraw
            # inflation adjust withdrawal next year
            withdraw = withdraw * (1 + inflation)
            results[s, t] = max(0.0, min(withdraw, balances[t] + withdraw))  # cashflow: can't exceed what's available
        # note: balances may go negative; cashflows clipped
    return results  # sims x years

# ---------------------- Simulation core ----------------------
@app.post('/simulate')
def simulate(req: SimRequest):
    user = req.user
    s = req.settings
    start_age = user.age
    retire_age = user.expected_retirement_age
    pot = user.accrued_pot
    sims = s.simulations
    max_age = s.max_age
    mu = s.mu
    sigma = s.sigma
    inflation = s.inflation

    years = list(range(retire_age, max_age + 1))
    n_years = len(years)

    # 1) Lump-sum invested + systematic withdrawals (4% rule baseline)
    withdrawal_rate = 0.04 if user.risk_pref != 'Aggressive' else 0.05
    lump_cashflows = simulate_withdrawals(pot, years, mu, sigma, inflation, withdrawal_rate, sims)

    # 2) Immediate life annuity (price annuity using approximate annuity factor)
    # assume annuity_rate increases with delay age: simple heuristic
    annuity_rate = 0.05 + 0.0005 * max(0, retire_age - 60)  # small bump for later claiming
    ann_payment = price_annuity(pot, annuity_rate)
    # annuity cashflows are deterministic until death; we simulate for each lifespan
    death_ages = sample_age_of_death(start_age, sims, max_age)
    ann_cashflows = np.zeros((sims, n_years))
    for i in range(sims):
        death_age = death_ages[i]
        # pay ann_payment for years where age <= death_age
        for yi, year in enumerate(years):
            if year <= death_age:
                ann_cashflows[i, yi] = ann_payment

    # 3) Hybrid: buy an annuity that covers floor (e.g., 50% of desired income), invest remainder
    desired_income = (user.current_salary or 100000) * (user.desired_income_pct or 70.0) / 100.0
    floor = 0.5 * desired_income
    annuity_needed = floor / annuity_rate if annuity_rate > 0 else 0
    annuity_needed = min(annuity_needed, pot)
    residual_pot = max(0.0, pot - annuity_needed)
    hybrid_ann_payment = price_annuity(annuity_needed, annuity_rate)
    hybrid_withdrawals = simulate_withdrawals(residual_pot, years, mu, sigma, inflation, withdrawal_rate, sims)
    hybrid_cashflows = hybrid_withdrawals + hybrid_ann_payment  # broadcast addition

    # Post-tax adjustments and aggregation per strategy
    def post_tax_summary(cf_matrix):
        # cf_matrix: sims x years
        # compute inflation-adjusted (real) cashflows using expected inflation
        # for simplicity, convert nominal cashflows to real by dividing by (1+inflation)^(t)
        t_idx = np.arange(n_years)
        real_cf = cf_matrix / ((1 + inflation) ** t_idx)
        # compute annual after-tax assuming tax on income
        after_tax = np.zeros_like(real_cf)
        taxes = np.zeros_like(real_cf)
        for i in range(real_cf.shape[0]):
            for t in range(real_cf.shape[1]):
                inc = real_cf[i, t]
                tax = simple_tax(inc, country=user.country)
                taxes[i, t] = tax
                after_tax[i, t] = inc - tax
        # metrics
        expected_real_income = after_tax.mean(axis=0).sum()
        median_path = np.median(after_tax, axis=0)
        p10 = np.percentile(after_tax, 10, axis=0)
        p90 = np.percentile(after_tax, 90, axis=0)
        # probability of ruin: fraction of sims with any year where balance was 0 for all remaining years
        # approximate: check if cumulative remaining nominal cashflows are zero sometime (for withdrawals sims)
        prob_ruin = np.mean((cf_matrix.sum(axis=1) == 0).astype(float))
        return {
            'expected_real_income_total': float(expected_real_income),
            'median_path': median_path.tolist(),
            'p10': p10.tolist(),
            'p90': p90.tolist(),
            'prob_ruin': float(prob_ruin)
        }

    lump_summary = post_tax_summary(lump_cashflows)
    ann_summary = post_tax_summary(ann_cashflows)
    hybrid_summary = post_tax_summary(hybrid_cashflows)

    # Simple scoring
    def score_from_summary(summary):
        # combine expected income and prob_ruin
        e = summary['expected_real_income_total']
        r = summary['prob_ruin']
        # naive normalization by pot to keep scale-friendly
        score = (e / (pot + 1e-9)) * 0.7 + (1 - r) * 0.3
        # scale to 0-100 roughly
        return float(max(0.0, min(100.0, score * 100)))

    resp = {
        'years': years,
        'strategies': {
            'lump_invested': {
                'summary': lump_summary,
                'score': score_from_summary(lump_summary),
            },
            'immediate_annuity': {
                'summary': ann_summary,
                'score': score_from_summary(ann_summary),
                'annuity_payment_nominal': ann_payment
            },
            'hybrid': {
                'summary': hybrid_summary,
                'score': score_from_summary(hybrid_summary),
                'annuity_floor': hybrid_ann_payment
            }
        },
        'meta': {
            'pot': pot,
            'retire_age': retire_age,
            'simulations': sims,
            'assumptions': {'mu': mu, 'sigma': sigma, 'inflation': inflation}
        }
    }
    return resp

# ---------------------- Sample dataset (for demo) ----------------------
SAMPLE_CSV = '''user_id,age,gender,current_salary,years_service,pension_type,accrued_pot,expected_retirement_age,marital_status,risk_pref,desired_income_pct,country
u001,60,F,80000,30,DefinedContribution,200000,62,Single,Conservative,60,India
u002,55,M,120000,33,DefinedContribution,450000,65,Married,Moderate,75,India
u003,50,M,100000,20,DefinedContribution,100000,60,Aggressive,50,India
'''

@app.get('/sample_dataset')
def get_sample_dataset():
    return {'csv': SAMPLE_CSV}

# ---------------------- Quick test runner (if executed directly) ----------------------
if __name__ == '__main__':
    print('This file defines a FastAPI app. Run with: uvicorn PensionOptimizer_prototype:app --reload')
