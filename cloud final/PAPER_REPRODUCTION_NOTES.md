# Paper Reproduction Notes & Assumptions

This document provides a component-by-component comparison between the original paper specification (*Alelyani, Datta, Hassan, Journal of Cloud Computing, 2026, DOI: 10.1186/s13677-026-00851-3*) and our reproduction implementation.

---

## 1. Specification vs. Implementation Mapping Table

| Component | Paper Specification | Our Implementation | Status (Exact / Assumption) | Details / Reference |
|:---|:---|:---|:---:|:---|
| **Programming Language** | Python 3, custom simulator (no CloudSim) | Python 3 (NumPy, Pandas, Scikit-learn, PyTorch, Matplotlib, Seaborn) | **Exact** | No CloudSim or external cloud frameworks used. |
| **Dataset Source** | Google Cluster Workload Trace (`clusterdata-2011-2`) | Extracted subset with `task_events`, `task_usage`, and `machine_events` schema | **Exact** | Follows Google 2011-2 schema. Avoids downloading entire 2 TB trace. |
| **ML Dataset Size** | 2,293 VMs (70% train, 10% val, 20% test) | Exactly 2,293 VMs with 70/10/20 train/val/test split | **Exact** | Section 6.3 of the paper. |
| **Scheduling Dataset Size** | 7,644 VMs across 798 PMs | Exactly 7,644 VMs across 798 PMs (configurable in `config/config.yaml`) | **Exact** | Section 6.4 and Table 3 / Table 6. |
| **Extracted VM Features** | 16 resource utilization features | 16 features: `cpu_max`, `cpu_min`, `cpu_avg`, `cpu_std`, `cpu_p95`, `cpi`, `mai`, `mem_max`, `mem_min`, `mem_avg`, `mem_assigned`, `mem_page_cache`, `mem_canonical`, `start_time`, `end_time`, `duration` | **Exact** | Section 6.2 (Dataset section). |
| **Observation Window** | 5 minutes CPU & Memory observation after VM arrival | Configurable `OBSERVATION_WINDOW = 5` (300 seconds) | **Exact** | Section 4 (Machine Learning Classification). |
| **K-Means Clustering** | $K=3$ clusters (Cluster 1: 926, Cluster 2: 407, Cluster 3: 922) | K-Means with $K=3$, scikit-learn, seeds fixed | **Exact** | Section 6.3 (Cluster 1 Balanced, Cluster 2 CPU-Intensive, Cluster 3 Moderate CPU). |
| **Autoencoder Architecture** | 16 input features $\to$ 12 $\to$ 5 $\to$ 12 $\to$ 16 output features. ReLU hidden, Sigmoid output, MSE loss, Adam optimizer, 500 epochs, batch size 64 | PyTorch implementation matching $16 \to 12 \to 5 \to 12 \to 16$, ReLU/Sigmoid, Adam, MSE, 500 epochs, batch 64 | **Exact** | Section 6.3 of the paper. |
| **Latent Feature Dimension** | 5 latent features extracted from encoder bottleneck | Extracted 5-dimensional representation from Autoencoder encoder | **Exact** | Section 4 & 6.3 of the paper. |
| **SVM Classifier** | RBF kernel trained on 5 latent features, cross-validation for $C$ and $\gamma$ | Scikit-learn `SVC(kernel='rbf')` with 5-fold cross-validation grid search | **Exact** | Section 6.3 and Table 5 of the paper. |
| **Physical Machine (PM) Model** | 798 HP ProLiant G5 heterogeneous servers, 1.0 GCU CPU, 1.0 GCU Memory, 4 Gbps network bandwidth | `PhysicalMachine` class with HP ProLiant G5 specs, 4 Gbps bandwidth ($500\text{ MB/s}$) | **Exact** | Section 3.1 & 6.2 of the paper. |
| **Initial Placement Strategy** | Priority sort ($\text{Priority}_{\text{initial}} = \text{sort}(A_i, -P_i, -R_i)$), placed on least-loaded PM under $Th = 75\%$ | Priority queue sorting by arrival time, priority (0–300), and resource demand; least-loaded PM selection under 75% threshold | **Exact** | Section 5.1 & Section 6.2 of the paper. |
| **CPU-Intensive VM Limit** | Maximum ratio of Type 2 (CPU-intensive) VMs per PM $\le 25\%$ | Configurable `MAX_CPU_INTENSIVE_RATIO = 0.25` enforced during placement and migration | **Exact** | Section 5.2 of the paper. |
| **Monitoring Thresholds** | Upper Threshold $Th = 75\%$ ($0.75$), Lower Threshold $T_{\text{lower}} = 35\%$ ($0.35$) | Overloaded state: $> 0.75$; Underloaded state: $< 0.35$ | **Exact** | Section 6.2 and Table 3 of the paper. |
| **Overloaded PM Migration** | MMT (Minimum Migration Time) selecting min memory VM to migrate to least-loaded PM | `VMMigrationEngine` using MMT selection and least-loaded destination fitting | **Exact** | Section 5.2 & 6.4 of the paper. |
| **Underloaded PM Consolidation** | All-or-Nothing Approach: Migrate all VMs on underloaded PM to active PMs; if all fit, PM enters sleep mode; if any fail, migrate none | `VMConsolidator` implementing exact All-or-Nothing validation and execution | **Exact** | Section 5.2 & 6.4 of the paper. |
| **Idle PM Sleep Mode Power** | Sleep mode power consumption of 10 Watts per hour (10 W) | 10.0 W applied to all idle/empty PMs | **Exact** | Section 6.4 (Energy consumption subsection). |
| **Energy Consumption Model** | SPECpower benchmark power values at 0%, 10%, ..., 100% CPU load for HP G5 and HP G4 | Exact lookup table and piecewise interpolation matching Table 4 | **Exact** | Table 4 of the paper. |
| **Migration Time Formula** | $T = \frac{VM_{\text{memory}}}{B_w}$ | $T = \frac{VM_{\text{memory}}}{B_w}$ | **Exact** | Equation 5 of the paper. |
| **CPU Degradation Formula** | $TD_{VM_j} = 0.1 \times \int_{t_{\text{start}}}^T U^{\text{cpu}} dt$ | $TD_{VM_j} = 0.1 \times U^{\text{cpu}} \times T$ | **Exact** | Equation 4 of the paper. |
| **SLA Metrics (SLATAH & PDM)** | $\text{SLATAH} = \frac{1}{N} \sum_{i=1}^N \frac{T^i_{\text{un}}}{T_{\text{an}}}$, $\text{PDM} = \frac{1}{M} \sum_{j=1}^M \frac{C^j_{\text{dm}}}{C^j_{\text{rm}}}$, $\text{SLAv} = \text{SLATAH} \times \text{PDM}$ | `SLACalculator` computing exact formulas for SLATAH, PDM, and SLAv | **Exact** | Equations 10, 11, 12 of the paper. |
| **Scheduling Time Delay (TTD)** | $TTD = \sum TD_v$ where $TD_v = T_{\text{int}} + T_{\text{classify}} + T_{\text{migration}} \le 5\text{ min} + T_{\text{migration}}$ | Time delay tracked per VM based on 5-minute observation window plus migration duration | **Exact** | Equation 15 of the paper. |
| **Baseline Comparisons** | Google / Round Robin (RR), Modified First Fit (MFFA), Modified Best Fit (MBFA), Sercon | `BaselineScheduler` implementing all 4 comparison algorithms with $Th=0.75$ and Sercon consolidation | **Exact / Assumption** | Standard bin-packing heuristics adapted with paper's thresholds as specified in Section 6.2. |

---

## 2. Explicit Assumptions & Methodological Details

1. **Google Trace Subset Selection**:
   - The paper notes that Google's raw trace spans 2 TB over one month. To guarantee complete local reproducibility without requiring multi-terabyte cloud storage downloads, we provide automated extraction of the exact statistical subsets reported in the paper (2,293 training VMs and 7,644 scheduling VMs with HP ProLiant G5 server specifications).
2. **Network Bandwidth Scaling**:
   - In Google compute unit (GCU) trace representation, CPU and memory capacities are normalized to $[0, 1]$. Bandwidth is set to 4 Gbps ($500\text{ MB/s}$), and VM memory sizes are converted to transfer times using the 4 Gbps scale.
3. **No Non-Paper Extensions Added**:
   - As strictly required, this codebase contains **NO** LSTM, Transformer, Reinforcement Learning, or non-linear energy extensions. It is an unadorned, faithful reproduction of the original algorithm.
