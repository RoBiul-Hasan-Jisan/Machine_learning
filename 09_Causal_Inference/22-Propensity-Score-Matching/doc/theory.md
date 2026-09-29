# 22. Propensity Score Matching

## Learning Objectives

- Describe 1:1 nearest-neighbor matching on the propensity score, with and without replacement
- Explain what estimand matching typically targets (ATT) and why
- Use calipers and balance checks, and know matching's main criticisms
- Implement matching and verify balance improves

---

## 1. The Problem

Weighting can produce extreme weights. **Matching** instead builds a comparison group by pairing each treated unit with the control unit that looks most similar on the propensity score.

## 2. The Concept

### 2.1 The estimator

For each treated unit $i$ find $j(i)=\arg\min_{j:T_j=0}|\,\text{logit}\,\hat e_i-\text{logit}\,\hat e_j|$. Then
$$\widehat{ATT}=\frac1{n_1}\sum_{i:T_i=1}\big(Y_i-Y_{j(i)}\big).$$
Matching on the **logit** of the score spreads out values near 0 and 1.

### 2.2 Why ATT

Every treated unit is retained and matched to a control, so the estimator answers "what was the effect on those who were treated?" Controls without a close treated match are simply dropped. Estimating the ATE needs matching for controls too (or weighting).

### 2.3 Design choices

- **With replacement** lowers bias (better matches) but lets one control be reused, which raises variance and makes naive standard errors wrong.
- **Caliper** (commonly 0.2 SD of the logit score) discards treated units with no close match, trading generalizability for bias.
- **Ratio:** 1:k matching uses more controls per treated unit.
- After matching, **re-check balance**; the goal is balanced covariates, not a particular $p$-value.

### 2.4 Criticisms

King and Nielsen (2019) argue that pruning on the propensity score can *increase* imbalance and model dependence compared with other matching metrics or weighting. Treat PSM as one option, and always report post-matching balance.

## 3. Use It: Code

`code/psm_demo.py` matches on the logit score with a caliper, reports the number of matched treated units, SMD before/after matching, and the ATT estimate (true ATT = 2).

## Exercises

1. Change the caliper to 0.05 SD. How many treated units are dropped?
2. Match without replacement (greedy). Compare bias.
3. Why can't you compute a valid standard error by treating matched pairs as independent when matching with replacement?

## Key Terms

| Term | What it actually means |
|---|---|
| Nearest-neighbor matching | Pair each treated unit with the closest control on the score |
| Caliper | Maximum allowed distance for a match |
| ATT | Average effect among the treated |
