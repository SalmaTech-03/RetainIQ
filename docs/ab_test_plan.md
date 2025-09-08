# A/B Test Plan: ChurnBuster Retention Campaign

## 1. Objective
The primary objective of this A/B test is to determine if a targeted retention offer can significantly reduce the churn rate for customers identified as "high-risk" by our predictive model. We will measure the impact on both customer retention and net retained revenue.

## 2. Population & Segments
- **Target Population:** All customers who are scored by the model and fall into the "High" risk band (predicted churn probability > 0.5).
- **Exclusion Criteria:**
    - Customers who have received a promotional offer in the last 90 days.
    - Customers who have explicitly opted out of marketing communications.
    - New customers with a tenure of less than 30 days.

## 3. Test Arms
The eligible population will be randomly assigned to one of two groups:

- **Arm A: Control Group (50% of users)**
    - Receives no special offer or communication. Their experience remains the same. This group serves as our baseline.

- **Arm B: Treatment Group (50% of users)**
    - Receives a targeted offer via email, for example: "We value your business! Enjoy 20% off your next month's bill as a thank you."

## 4. Metrics
- **Primary Metric:** Churn Rate within 60 days of receiving the offer. Churn is defined as the customer deactivating their service.
- **Secondary Metrics:**
    - **Net Retained Revenue:** (Revenue from retained customers in Treatment) - (Cost of offers).
    - **Offer Acceptance Rate:** Percentage of users in the Treatment group who accept the offer.

## 5. Sample Size Calculation
Based on our analysis in the `03_ab_test_design.ipynb` notebook:
- **Baseline Churn Rate (for high-risk users):** 50%
- **Minimum Detectable Effect (MDE):** 5% (we want to detect a drop in churn from 50% to 45%)
- **Statistical Power:** 80%
- **Significance Level (Alpha):** 5%

**Required Sample Size:** We need approximately **1565 customers per group** (1565 in Control, 1565 in Treatment) to achieve statistically significant results.