# Dataset & Feature Dictionary
## UCI Garment Worker Productivity Dataset (Dhaka Industrial Plants)

### 1. Raw Dataset Attributes
| Column Name | Raw Dtype | Description | Physical Unit | Missing Count | Handling Strategy |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `date` | `object` | Date of shift record | MM/DD/YYYY | 0 | Converted to datetime object; extracted temporal calendar features. |
| `quarter` | `object` | Month quarter breakdown | Quarter1 - Quarter5 | 0 | One-hot encoded. |
| `department` | `object` | Department line name | 'sweing', 'finishing', 'finishing ' | 0 | Whitespace trimmed, typo corrected to 'sewing'. One-hot encoded. |
| `day` | `object` | Day of the working week | Sunday - Saturday (no Friday) | 0 | One-hot encoded. Reflects 6-day Bangladesh work week. |
| `team` | `int64` | Identifier of the production team | 1 to 12 | 0 | Converted to categorical string and one-hot encoded. |
| `targeted_productivity` | `float64` | Target productivity quota set by authority | Ratio [0.07, 0.80] | 0 | Standardized numerical predictor. |
| `smv` | `float64` | Standard Minute Value (allocated time for task) | Minutes per garment style | 0 | Standardized numerical predictor. Reflects garment complexity. |
| `wip` | `float64` | Work in progress (partially completed garments) | Garment count | 506 | 100% of missing values belong to finishing. Imputed 0 for finishing, median for sewing, followed by log1p. |
| `over_time` | `int64` | Total overtime across entire team | Minutes | 0 | Anomaly on 3/9/2015 fixed. Normalized per worker. |
| `incentive` | `int64` | Financial incentive allocated to team | Bangladeshi Taka (BDT) | 0 | Anomaly on 3/9/2015 fixed. Normalized per worker. |
| `idle_time` | `float64` | Duration of production interruption | Minutes | 0 | Standardized numerical predictor. Binary indicator also engineered. |
| `idle_men` | `int64` | Number of idle workers during stoppage | Headcount | 0 | Standardized numerical predictor. |
| `no_of_style_change` | `int64` | Number of style transitions in shift | Count (0, 1, 2) | 0 | Standardized numerical predictor + binary indicator. |
| `no_of_workers` | `float64` | Total headcount of workers assigned to team | Headcount [2, 89] | 0 | Standardized numerical predictor. Used to normalize overtime and incentive. |
| `actual_productivity` | `float64` | Realized productivity ratio | Ratio [0.23, 1.12] | 0 | **Primary Continuous Target (Regression)**. |

---

### 2. Engineered Features
| Feature Name | Computation Formula | Operational Rationale |
| :--- | :--- | :--- |
| `target_met` | `(actual_productivity >= targeted_productivity).astype(int)` | **Primary Binary Target (Classification)**. |
| `productivity_gap` | `actual_productivity - targeted_productivity` | Performance margin evaluation. |
| `overtime_per_worker` | `over_time / max(no_of_workers, 1)` | Standardizes team-level minutes across varying team sizes (8 vs 60 workers). |
| `incentive_per_worker` | `incentive / max(no_of_workers, 1)` | Incentive density per worker. |
| `smv_per_worker` | `smv / max(no_of_workers, 1)` | Individual task complexity burden. |
| `workload_intensity` | `smv * targeted_productivity` | Total standard minute output expected per unit time. |
| `overtime_fatigue_penalty` | `max(0, overtime_per_worker - 120)` | Captures non-linear physiological exhaustion past 2 hours of daily overtime. |
| `log_wip` | `log(1 + wip_clean)` | Stabilizes variance and handles right-skewness of inventory buffer. |
| `has_idle_time` | `(idle_time > 0).astype(int)` | Binary flag for machine stoppage / breakdown. |
| `has_style_change` | `(no_of_style_change > 0).astype(int)` | Binary flag for disruption caused by garment style changes. |
| `day_of_month` | `date.dt.day` | Captures intra-month order delivery cycles. |
| `week_of_year` | `date.dt.isocalendar().week` | Captures seasonal factory pace. |
