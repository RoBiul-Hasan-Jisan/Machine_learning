# 24. ATE, ATT and ATC

## Learning Objectives

- Define ATE, ATT and ATC and explain when each differs from the others
- Write the g-computation and IPW estimators for each estimand
- Show numerically that choosing the wrong estimand gives a different (still "correct") number for a different question
- Explain which estimands need less overlap

---

## 1. The Problem

Module 05 introduced ATE, ATT and CATE. Now that we have estimators (Modules 19–23), we need to say **which population** each one averages over, because with effect heterogeneity the answers differ.

## 2. The Concept

### 2.1 Three estimands

$$ATE=E[Y(1)-Y(0)],\quad ATT=E[Y(1)-Y(0)\mid T=1],\quad ATC=E[Y(1)-Y(0)\mid T=0],$$
and $ATE=P(T{=}1)\,ATT+P(T{=}0)\,ATC$. If people who benefit more select into treatment, $ATT>ATE>ATC$.

### 2.2 Which question?

| Question | Estimand |
|---|---|
| Effect of a universal policy | ATE |
| Was the program worth it for the people it reached? | ATT |
| What would happen if we extended it to those not treated? | ATC |

### 2.3 Estimators

**G-computation:** fit $\hat m_t(x)$; average $\hat m_1(X_i)-\hat m_0(X_i)$ over *all* units (ATE), treated units (ATT) or control units (ATC).

**IPW weights:**

| Estimand | Treated weight | Control weight |
|---|---|---|
| ATE | $1/e$ | $1/(1-e)$ |
| ATT | $1$ | $e/(1-e)$ |
| ATC | $(1-e)/e$ | $1$ |

For the ATT, treated units are left as they are and controls are re-weighted to look like them.

### 2.4 Overlap needs differ

ATT needs every treated unit to have comparable controls ($e<1$), but does *not* require comparable treated units for every control. ATC needs the reverse. ATE needs both.

## 3. Use It: Code

`code/estimands_demo.py` simulates heterogeneous effects $\tau(x)=1+2x_0$ with treatment more likely for high $x_0$, computes the true ATE/ATT/ATC from the potential outcomes, and estimates each with g-computation and IPW.

## Exercises

1. Reverse the selection ($T$ more likely for low $x_0$). What happens to the ordering?
2. Show $ATE=P(T{=}1)ATT+P(T{=}0)ATC$ in the simulation.
3. Which estimand would you report for a job-training program open only to volunteers?

## Key Terms

| Term | What it actually means |
|---|---|
| ATT | Effect among those actually treated |
| ATC | Effect among those not treated (would-be effect of treating them) |
| ATT weights | 1 for treated, $e/(1-e)$ for controls |
