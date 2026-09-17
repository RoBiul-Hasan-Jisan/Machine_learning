# Naive Bayes Classification Questions

## Question 1 — Multinomial Naive Bayes: Inbox Spam Filter

EduTrack's mail server logged the following 6 emails (word lists) received by students:

| Email ID | Words                       | Category |
| -------- | --------------------------- | -------- |
| E1       | Free, Prize, Win, Claim     | Spam     |
| E2       | Meeting, Project, Schedule  | Not Spam |
| E3       | Free, Win, Prize, Lottery   | Spam     |
| E4       | Discussion, Agenda, Meeting | Not Spam |
| E5       | Claim, Free, Offer, Prize   | Spam     |
| E6       | Project, Report, Deadline   | Not Spam |

### New Email to Classify

> **{Free, Meeting, Prize, Claim}**

### Tasks

**(a)** Compute the prior probabilities:

* $P(\text{Spam})$
* $P(\text{Not Spam})$

**(b)** Build the vocabulary and count each word's frequency within each class.

**(c)** Using **Laplace (add-1) smoothing**, compute the likelihood of the new email under each class.

**(d)** Compare the two posterior scores and classify the email as **Spam** or **Not Spam**.

---

## Question 2 — Bernoulli Naive Bayes: Cafeteria Review Sentiment

EduTrack's cafeteria feedback kiosk logged the following 6 short reviews (keywords present in each review):

| Review ID | Keywords Present             | Sentiment |
| --------- | ---------------------------- | --------- |
| R1        | tasty, clean, good, friendly | Positive  |
| R2        | slow, dirty, rude, bad       | Negative  |
| R3        | clean, fast, tasty, good     | Positive  |
| R4        | rude, bad, dirty, slow       | Negative  |
| R5        | clean, good, friendly, tasty | Positive  |
| R6        | dirty, bland, slow, bad      | Negative  |

### New Review to Classify

> **{good, tasty, rude, clean}**

### Tasks

**(a)** List the full vocabulary (**10 distinct keywords across all reviews**) and build a binary presence/absence **(1/0) matrix** for the 6 reviews.

**(b)** Compute the prior probabilities:

* $P(\text{Positive})$
* $P(\text{Negative})$

**(c)** Using **Laplace smoothing**, compute:

* $P(\text{review} \mid \text{Positive})$
* $P(\text{review} \mid \text{Negative})$

> **Important:** Remember to include the probability terms for keywords that are **absent** from the new review.

**(d)** Classify the new review as **Positive** or **Negative**.

---

## Question 3 — Gaussian Naive Bayes: Flu Diagnosis

The campus health center recorded vitals for 6 recent visits:

| Patient | Temperature (°C) | Heart Rate (bpm) | Class  |
| ------- | ---------------: | ---------------: | ------ |
| P1      |             38.5 |              110 | Flu    |
| P2      |             39.0 |              105 | Flu    |
| P3      |             38.0 |              108 | Flu    |
| P4      |             36.5 |               80 | No Flu |
| P5      |             36.8 |               75 | No Flu |
| P6      |             37.0 |               78 | No Flu |

### New Patient to Diagnose

> **Temperature = 37.6°C**
> **Heart Rate = 90 bpm**

### Tasks

**(a)** Compute the **mean and standard deviation** of Temperature and Heart Rate separately for:

* Flu class
* No Flu class

**(b)** Using the **Gaussian probability density function (PDF)**, compute the likelihood of the new patient's:

* Temperature under each class
* Heart Rate under each class

**(c)** Combine these likelihoods with the class priors to get a **posterior score** for:

* Flu
* No Flu

Then diagnose the patient as **Flu** or **No Flu**.
