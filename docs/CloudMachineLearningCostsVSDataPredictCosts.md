# [High Value Project Tutorials](HighValueProjectTutorials.md) - Cloud Machine Learning Costs VS DataPredict Costs

## General Pricing

| Factor                       | DataPredict™ | Google Cloud | Microsoft Azure | Amazon SageMaker |
|------------------------------|--------------|--------------|-----------------|------------------|
| Lowest Cost Per Hour Per CPU | $0           | $0.22        | $0.113          | $0.050           |
| Lowest Cost Per Storage GB   | $0           | $0.22        | $0.15           | $0.4             |
| Certification Cost Per Exam  | N/A          | $200         | $99-$165        | $100-$300        |

* Information based on 25 June 2026.

## Scalability

| Factor                               | DataPredict™                                             | Google Cloud               | Microsoft Azure            | Amazon SageMaker           |
|--------------------------------------|----------------------------------------------------------|----------------------------|----------------------------|----------------------------|
| Player Count Per CPU                 | Free CPU Per Player                                      | Must Purchase Individually | Must Purchase Individually | Must Purchase Individually |
| Can Each Player Run Their Own Models | Yes (Free CPU Per Player)                                | Yes (Paid CPU Per Player)  | Yes (Paid CPU Per Player)  | Yes (Paid CPU Per Player)  |
| Can Share GPU                        | Cannot Use GPU, But Redundant Due To Free CPU Per Player | Yes                        | Yes                        | Yes                        |

## Total Monthly Player Count Cost

The table assumes that:

* The game has an average play time of 10 minutes.

* A single machine learning model is required to perform constant training and predictions.

* Single CPU only.

Note: Not to be confused with concurrent users. 

| Total Monthly Player Count | DataPredict™ | Google Cloud               | Microsoft Azure            | Amazon SageMaker           |
|----------------------------|--------------|----------------------------|----------------------------|----------------------------|
| 10                         | $0           | $3.667                     | $1.883                     | $0.833                     |
| 100                        | $0           | $36.667                    | $18.833                    | $8.333                     |
| 1000                       | $0           | $366.667                   | $188.333                   | $83.333                    |
| 10000                      | $0           | $3666.667                  | $1883.333                  | $883.333                   |
| 100000                     | $0           | $36666.667                 | $18833.333                 | $8833.333                  |

## DataPredict™ License Cost

| License Type Based On Gross Revenue Per Company / Organization / Group / Individual | Percentage To Pay Based On Project's Gross Revenue In USD | Additional Notes                          | Licensing Cost When A Project Earns 5K / 10K / 100K / 1M USD Per Month |
|-------------------------------------------------------------------------------------|-----------------------------------------------------------|-------------------------------------------|------------------------------------------------------------------------|
| Non-B2B + Less Than Or Equal To 5K USD Within 365 Days (Not Per 365 Days)           | 0%                                                        | Requires Public Disclosure Of DataPredict | Not Applicable                                                         |
| Standard                                                                            | 2% That Decreases As Gross Revenue Increases              | Requires Public Disclosure Of DataPredict | 89 USD / 160 USD / 1300 USD / 10,530 USD                               |
| White-Label                                                                         | 5%                                                        | None                                      | 250 USD / 500 USD / 5000 USD / 50,000 USD                              |

* Note: At the $5,000/year revenue mark, the licensing fee is approximately $8/month for "Standard" license. Collecting micro-payments of this size creates administrative overhead that exceeds the revenue itself. Therefore, we subsidize 100% of costs for organizations under this threshold. We only begin billing when your success generates a fee large enough to be meaningful for both parties.

## Full Cost & Profitability Comparison Per Player

The table assumes that:

* The game has an average play time of 10 minutes.

* A single machine learning model is required to perform constant training and predictions.

* Single CPU only.

| Metric                                    | DataPredict (Standard) | DataPredict (White-Label) | Google Cloud | Microsoft Azure | Amazon SageMaker |
|-------------------------------------------|------------------------|---------------------------|--------------|-----------------|------------------|
| Lowest Infrastructure Cost (CPU Only)     | $0                     | $0                        | $0.367       | $0.188          | $0.083           |
| ARPU (Average Revenue Per User)           | $0.50                  | $0.50                     | $0.50        | $0.50           | $0.50            |
| Monthly Gross Revenue Per Player          | $0.50                  | $0.50                     | $0.50        | $0.50           | $0.50            |
| ML Cost Per Player                        | $0.010                 | $0.025                    | $0.367       | $0.188          | $0.083           |
| Monthly Net Revenue Per Player (After ML) | $0.490                 | $0.475                    | $0.133       | $0.312          | $0.417           |
| Profit Margin Per Player (After ML)       | 98%                    | 95%                       | 27%          | 62%             | 82%              |
| ML Cost as % of Revenue Per Player        | 2%                     | 5%                        | 73%          | 38%             | 18%              |
