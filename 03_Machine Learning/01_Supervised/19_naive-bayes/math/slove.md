# Question 1 — Multinomial Naive Bayes: Inbox Spam Filter

EduTrack's mail server logged the following 6 emails:

| Email ID | Words                       | Category |
| -------- | --------------------------- | -------- |
| E1       | Free, Prize, Win, Claim     | Spam     |
| E2       | Meeting, Project, Schedule  | Not Spam |
| E3       | Free, Win, Prize, Lottery   | Spam     |
| E4       | Discussion, Agenda, Meeting | Not Spam |
| E5       | Claim, Free, Offer, Prize   | Spam     |
| E6       | Project, Report, Deadline   | Not Spam |

### New Email

**{Free, Meeting, Prize, Claim}**

---

## Prior Probabilities

There are 3 Spam emails and 3 Not Spam emails out of 6 total emails.

$$
P(\text{Spam}) = \frac{3}{6}=0.5
$$

$$
P(\text{Not Spam}) = \frac{3}{6}=0.5
$$

---

## Vocabulary and Word Counts per Class

### Vocabulary

There are **13 unique words**:

`Free, Prize, Win, Claim, Lottery, Offer, Meeting, Project, Schedule, Discussion, Agenda, Report, Deadline`

### Word Frequency

| Word             | Spam Count | Not Spam Count |
| ---------------- | ---------: | -------------: |
| Free             |          3 |              0 |
| Prize            |          3 |              0 |
| Win              |          2 |              0 |
| Claim            |          2 |              0 |
| Lottery          |          1 |              0 |
| Offer            |          1 |              0 |
| Meeting          |          0 |              2 |
| Project          |          0 |              2 |
| Schedule         |          0 |              1 |
| Discussion       |          0 |              1 |
| Agenda           |          0 |              1 |
| Report           |          0 |              1 |
| Deadline         |          0 |              1 |
| **Total Tokens** |     **12** |          **9** |

---

## Laplace-Smoothed Likelihoods

For Multinomial Naive Bayes, using Laplace (add-1) smoothing:

$$
P(w\mid c)=
\frac{\text{count}(w,c)+1}
{N_c+|V|}
$$

where:

* $N_c$ = total number of word tokens in class $c$
* $|V|$ = vocabulary size
* $|V|=13$

### Spam

$$
N_{\text{Spam}}=12
$$

Therefore:

$$
N_{\text{Spam}}+|V|=12+13=25
$$

### Not Spam

$$
N_{\text{Not Spam}}=9
$$

Therefore:

$$
N_{\text{Not Spam}}+|V|=9+13=22
$$

### Required Word Probabilities

The new email contains:

`Free, Meeting, Prize, Claim`

| Word    | $P(w\mid\text{Spam})$ | $P(w\mid\text{Not Spam})$ |
| ------- | --------------------: | ------------------------: |
| Free    |         $4/25=0.1600$ |             $1/22=0.0455$ |
| Meeting |         $1/25=0.0400$ |             $3/22=0.1364$ |
| Prize   |         $4/25=0.1600$ |             $1/22=0.0455$ |
| Claim   |         $3/25=0.1200$ |             $1/22=0.0455$ |

---

## Likelihood of the New Email

### Spam

$$
\begin{aligned}
P(\text{email}\mid\text{Spam})
&=
P(\text{Free}\mid\text{Spam})
\times
P(\text{Meeting}\mid\text{Spam})\\
&\quad\times
P(\text{Prize}\mid\text{Spam})
\times
P(\text{Claim}\mid\text{Spam})\\
&=
0.16\times0.04\times0.16\times0.12\\
&=
1.229\times10^{-4}
\end{aligned}
$$

### Not Spam

$$
\begin{aligned}
P(\text{email}\mid\text{Not Spam})
&=
P(\text{Free}\mid\text{Not Spam})
\times
P(\text{Meeting}\mid\text{Not Spam})\\
&\quad\times
P(\text{Prize}\mid\text{Not Spam})
\times
P(\text{Claim}\mid\text{Not Spam})\\
&=
0.0455\times0.1364\times0.0455\times0.0455\\
&\approx
1.28\times10^{-5}
\end{aligned}
$$

---

## Posterior Comparison

Using:

$$
\text{Score}(c)=
P(c)\times P(\text{email}\mid c)
$$

### Spam Score

$$
\begin{aligned}
\text{Score(Spam)}
&=
0.5\times1.229\times10^{-4}\\
&=
6.14\times10^{-5}
\end{aligned}
$$

### Not Spam Score

$$
\begin{aligned}
\text{Score(Not Spam)}
&=
0.5\times1.28\times10^{-5}\\
&=
6.40\times10^{-6}
\end{aligned}
$$

Comparison:

$$
6.14\times10^{-5}
\gg
6.40\times10^{-6}
$$

Therefore:

> **Classification: Spam**


## Question 2 — Bernoulli Naive Bayes: Cafeteria Review Sentiment

| Review ID | Keywords Present             | Sentiment |
| --------- | ---------------------------- | --------- |
| R1        | tasty, clean, good, friendly | Positive  |
| R2        | slow, dirty, rude, bad       | Negative  |
| R3        | clean, fast, tasty, good     | Positive  |
| R4        | rude, bad, dirty, slow       | Negative  |
| R5        | clean, good, friendly, tasty | Positive  |
| R6        | dirty, bland, slow, bad      | Negative  |

### New Review

**{good, tasty, rude, clean}**

### Vocabulary and Binary Presence Matrix

**Vocabulary (10 keywords):**

`tasty, clean, good, friendly, slow, dirty, rude, bad, fast, bland`

| Review | tasty | clean | good | friendly | slow | dirty | rude | bad | fast | bland |
| ------ | ----: | ----: | ---: | -------: | ---: | ----: | ---: | --: | ---: | ----: |
| R1     |     1 |     1 |    1 |        1 |    0 |     0 |    0 |   0 |    0 |     0 |
| R2     |     0 |     0 |    0 |        0 |    1 |     1 |    1 |   1 |    0 |     0 |
| R3     |     1 |     1 |    1 |        0 |    0 |     0 |    0 |   0 |    1 |     0 |
| R4     |     0 |     0 |    0 |        0 |    1 |     1 |    1 |   1 |    0 |     0 |
| R5     |     1 |     1 |    1 |        1 |    0 |     0 |    0 |   0 |    0 |     0 |
| R6     |     0 |     0 |    0 |        0 |    1 |     1 |    0 |   1 |    0 |     1 |

### Prior Probabilities

There are 3 positive and 3 negative reviews.

$$
P(\text{Positive}) = \frac{3}{6}=0.5
$$

$$
P(\text{Negative}) = \frac{3}{6}=0.5
$$

### Laplace-Smoothed Class-Conditional Probabilities

For Bernoulli Naive Bayes:

$$
P(w=1\mid c)=
\frac{\text{count}(w=1,c)+1}{N_c+2}
$$

where:

$$
N_c=3
$$

for each class.

Therefore, the denominator is:

$$
N_c+2=3+2=5
$$

| Word     | $P(w=1\mid\text{Positive})$ | $P(w=1\mid\text{Negative})$ |
| -------- | --------------------------: | --------------------------: |
| tasty    |                   $4/5=0.8$ |                   $1/5=0.2$ |
| clean    |                   $4/5=0.8$ |                   $1/5=0.2$ |
| good     |                   $4/5=0.8$ |                   $1/5=0.2$ |
| friendly |                   $3/5=0.6$ |                   $1/5=0.2$ |
| slow     |                   $1/5=0.2$ |                   $4/5=0.8$ |
| dirty    |                   $1/5=0.2$ |                   $4/5=0.8$ |
| rude     |                   $1/5=0.2$ |                   $3/5=0.6$ |
| bad      |                   $1/5=0.2$ |                   $4/5=0.8$ |
| fast     |                   $2/5=0.4$ |                   $1/5=0.2$ |
| bland    |                   $1/5=0.2$ |                   $2/5=0.4$ |

The new review contains:

**Present (1):**

`good, tasty, rude, clean`

**Absent (0):**

`friendly, slow, dirty, bad, fast, bland`

For absent words, we use:

$$
P(w=0\mid c)=1-P(w=1\mid c)
$$

### Likelihood — Positive

$$
\begin{aligned}
P(\text{review}\mid\text{Positive})
&=
(0.8\times0.8\times0.2\times0.8)\\
&\quad\times(0.4\times0.8\times0.8\times0.8\times0.6\times0.8)\\
&=0.01007
\end{aligned}
$$

The first group represents the **present** words:

`good, tasty, rude, clean`

The second group represents the **absent** words:

`friendly, slow, dirty, bad, fast, bland`

### Likelihood — Negative

$$
\begin{aligned}
P(\text{review}\mid\text{Negative})
&=
(0.2\times0.2\times0.6\times0.2)\\
&\quad\times(0.8\times0.2\times0.2\times0.2\times0.8\times0.6)\\
&=1.475\times10^{-5}
\end{aligned}
$$

### Classification

Positive posterior score:

$$
\text{Score(Positive)}=
P(\text{Positive})\times P(\text{review}\mid\text{Positive})
$$

$$
=0.5\times0.01007=
5.033\times10^{-3}
$$

Negative posterior score:

$$
\text{Score(Negative)}=
P(\text{Negative})\times P(\text{review}\mid\text{Negative})
$$

$$
=0.5\times1.475\times10^{-5}=
7.37\times10^{-6}
$$

Since:

$$
5.033\times10^{-3}
\gg
7.37\times10^{-6}
$$

Therefore:

> **Classification: Positive**

---

# Question 3 — Gaussian Naive Bayes: Flu Diagnosis

The campus health center recorded the following data:

| Patient | Temperature (°C) | Heart Rate (bpm) | Class  |
| ------- | ---------------: | ---------------: | ------ |
| P1      |             38.5 |              110 | Flu    |
| P2      |             39.0 |              105 | Flu    |
| P3      |             38.0 |              108 | Flu    |
| P4      |             36.5 |               80 | No Flu |
| P5      |             36.8 |               75 | No Flu |
| P6      |             37.0 |               78 | No Flu |

### New Patient

* **Temperature:** 37.6°C
* **Heart Rate:** 90 bpm

> **Note:** Sample standard deviation ($n-1$) is used below.

## Mean and Standard Deviation per Class

| Class  | $\mu_{\text{Temp}}$ | $\sigma_{\text{Temp}}$ | $\mu_{\text{HR}}$ | $\sigma_{\text{HR}}$ |
| ------ | ------------------: | ---------------------: | ----------------: | -------------------: |
| Flu    |              38.500 |                  0.500 |           107.667 |                2.517 |
| No Flu |              36.767 |                  0.252 |            77.667 |                2.517 |

## Gaussian Probability Density Function

The Gaussian PDF is:

$$
f(x)=
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$$

## Gaussian Likelihoods

For the new patient's temperature and heart rate:

| Feature               |                  Flu |              No Flu |
| --------------------- | -------------------: | ------------------: |
| $f(\text{Temp}=37.6)$ |             $0.1579$ |          $0.006597$ |
| $f(\text{HR}=90)$     | $3.21\times10^{-12}$ | $9.65\times10^{-7}$ |

The heart rate of **90 bpm** is approximately:

* 17.7 bpm away from the Flu mean (107.7 bpm)
* 12.3 bpm away from the No-Flu mean (77.7 bpm)

Because the standard deviations are small, the Gaussian likelihood for the Flu class becomes extremely small.

## Posterior Scores

The class priors are:

$$
P(\text{Flu})=\frac{3}{6}=0.5
$$

$$
P(\text{No Flu})=\frac{3}{6}=0.5
$$

### Flu Score

$$
\begin{aligned}
\text{Score(Flu)}
&=
P(\text{Flu})
\times
f(\text{Temp}\mid\text{Flu})
\times
f(\text{HR}\mid\text{Flu})\\
&=
0.5\times0.1579\times3.21\times10^{-12}\\
&\approx2.53\times10^{-13}
\end{aligned}
$$

### No Flu Score

$$
\begin{aligned}
\text{Score(No Flu)}
&=
P(\text{No Flu})
\times
f(\text{Temp}\mid\text{No Flu})
\times
f(\text{HR}\mid\text{No Flu})\\
&=
0.5\times0.006597\times9.65\times10^{-7}\\
&\approx3.18\times10^{-9}
\end{aligned}
$$

Since:

$$
3.18\times10^{-9}
\gg
2.53\times10^{-13}
$$

Therefore:

> **Diagnosis: No Flu**
