# Implementation Plan: ML-Based VM Placement in Data Centres

Reproducing the original research paper:
> **"A Machine Learning Technique for Optimizing Virtual Machine Placement in Data-Centres"**
> *Abdullah Alelyani, Amitava Datta, Ghulam Mubashar Hassan*
> *Journal of Cloud Computing, Vol. 15, Article 37 (2026)*
> DOI: [10.1186/s13677-026-00851-3](https://doi.org/10.1186/s13677-026-00851-3)

---

## 1. Algorithm Breakdown & Workflow

The paper proposes a two-stage scheduling algorithm with machine learning:
1. **Preliminary Scheduling Stage**:
   - Arriving VMs are sorted by priority: $\text{Priority}_{\text{initial}} = \text{sort}(\text{VMs}, \{A_i \uparrow, -P_i \downarrow, R_i \downarrow\})$.
   - Each VM is initially placed on the least-loaded Physical Machine (PM) in terms of both CPU and memory load under the upper threshold $Th = 75\%$.
   - The VM runs for a **5-minute observation window** (`OBSERVATION_WINDOW = 5`), during which time-series CPU and memory resource usage events are gathered.
   - 16 VM resource features are extracted from the 5-minute window.
   - The trained Autoencoder compresses the 16 features into **5 latent features**.
   - The trained SVM classifier predicts the VM class (Class 1: Balanced/Low CPU, Class 2: CPU-Intensive, Class 3: Moderate CPU/Low Memory).
2. **Consolidation & Migration Stage**:
   - PMs are categorized into:
     - $L_{m1}$: Overloaded PMs (CPU $> 75\%$ OR Memory $> 75\%$ OR CPU-Intensive Class 2 VM ratio $> 25\%$).
     - $L_{m2}$: Underloaded PMs (CPU $< 35\%$ AND Memory $< 35\%$).
   - **Overloaded PM Mitigation**: Candidate VMs are selected using MMT (Minimum Migration Time / minimum memory) and migrated to least-loaded PMs to maintain load $\le 75\%$ and Class 2 VM ratio $\le 25\%$.
   - **Underloaded PM Consolidation**: All-or-nothing approach — attempt to migrate all VMs on the underloaded PM to other active PMs. If all VMs can be placed without violating constraints, the VMs migrate and the underloaded PM is powered down into Sleep mode (10 W).
   - **Energy, SLA, and Resource Metrics** are computed continuously over time.

---

## 2. Dataset Requirements & 16-Feature Mapping

### 2.1 Google Cluster Workload Trace
Google cluster trace data schema (`cluster-data-2011-2`):
- `machine_events`: Machine ID, CPU capacity, Memory capacity.
- `task_events` / `instance_events`: Job ID, Task index, Machine ID, Event type, Priority (0–300), Resource requests.
- `task_usage`: Start time, End time, Mean CPU rate, Canonical memory usage, Assigned memory usage, Unmapped page cache, Total page cache, Max memory, CPI, MAI, Sample CPU percentiles.

### 2.2 16 Features Mapping Table

| Index | Feature Name | Description | Formula / Trace Field |
|:-----:|:-------------|:------------|:----------------------|
| 1 | `cpu_max` | Maximum CPU utilization | $\max(\text{CPU usage})$ in 5-min window |
| 2 | `cpu_min` | Minimum CPU utilization | $\min(\text{CPU usage})$ in 5-min window |
| 3 | `cpu_avg` | Mean CPU utilization | $\text{mean}(\text{CPU usage})$ in 5-min window |
| 4 | `cpu_std` | CPU usage standard deviation | $\text{std}(\text{CPU usage})$ in 5-min window |
| 5 | `cpu_p95` | Peak CPU demand | 95th percentile CPU usage in window |
| 6 | `cpi` | Cycles Per Instruction | Google `task_usage.cpi` |
| 7 | `mai` | Memory Accesses Per Instruction | Google `task_usage.mai` |
| 8 | `mem_max` | Maximum memory utilization | $\max(\text{Memory usage})$ in window |
| 9 | `mem_min` | Minimum memory utilization | $\min(\text{Memory usage})$ in window |
| 10 | `mem_avg` | Average memory utilization | $\text{mean}(\text{Memory usage})$ in window |
| 11 | `mem_assigned` | Assigned memory | Google `task_usage.assigned_memory` |
| 12 | `mem_page_cache` | Page cache memory | Google `task_usage.page_cache_memory` |
| 13 | `mem_canonical` | Canonical memory usage | Google `task_usage.canonical_memory` |
| 14 | `start_time` | VM / Job start timestamp | Google `task_events.timestamp` (start) |
| 15 | `end_time` | VM / Job end timestamp | Google `task_events.timestamp` (end) |
| 16 | `duration` | Total runtime window | $\text{end\_time} - \text{start\_time}$ |

---

## 3. ML Pipeline Specification

1. **K-Means Clustering**:
   - $K=3$ clusters.
   - Normalized 16 features.
   - Cluster 1: Balanced / Low CPU (mean CPU $\approx 0.028$). Paper training count: 926.
   - Cluster 2: CPU-Intensive (mean CPU $\approx 0.17$, 10x cluster 1). Paper training count: 407.
   - Cluster 3: Moderate CPU / Low Memory (mean CPU $\approx 0.0542$). Paper training count: 922.
2. **Autoencoder (PyTorch)**:
   - Architecture: $16 \to 12 \to 5 \to 12 \to 16$.
   - Encoder: Dense(16, 12), ReLU $\to$ Dense(12, 5), ReLU.
   - Decoder: Dense(5, 12), ReLU $\to$ Dense(12, 16), Sigmoid.
   - Optimizer: Adam, Loss: MSE, Epochs: 500, Batch size: 64.
   - Extracts 5 latent features.
3. **SVM Classifier (Scikit-Learn)**:
   - Input: 5 latent features. Target: K-Means cluster labels.
   - Kernel: RBF with cross-validation on $C$ and $\gamma$.
   - Metrics: Accuracy ($\approx 77\%$), Precision ($\approx 79\%$), Recall ($\approx 80\%$), F1 ($\approx 0.77$).

---

## 4. Cloud Simulation & Mathematical Models

### 4.1 Physical Machine Model (HP ProLiant G5)
- Heterogeneous PMs ($N \approx 798 / 800$).
- Network bandwidth $B_w = 4\text{ Gbps}$ (500 MB/s).
- SPECpower HP G5 Energy Table:
  - Sleep: 10 W
  - 0%: 93.7 W, 10%: 97.0 W, 20%: 101.0 W, 30%: 105.0 W, 40%: 110.0 W
  - 50%: 116.0 W, 60%: 121.0 W, 70%: 125.0 W, 80%: 129.0 W, 90%: 133.0 W, 100%: 135.0 W

### 4.2 Migration & SLA Formulas
- **Migration Time**: $T = \frac{\text{VM}_{\text{memory}}}{B_w}$.
- **CPU Degradation**: $TD_{\text{VM}_j} = 0.1 \times \int_{t_{\text{start}}}^{T} U^{\text{cpu}} dt$.
- **SLATAH**: $\text{SLATAH} = \frac{1}{N} \sum_{i=1}^N \frac{T^i_{\text{un}}}{T_{\text{an}}}$.
- **PDM**: $\text{PDM} = \frac{1}{M} \sum_{j=1}^M \frac{C^j_{\text{dm}}}{C^j_{\text{rm}}}$.
- **Overall SLAv**: $\text{SLAv} = \text{SLATAH} \times \text{PDM}$.
- **Total Time Delay (TTD)**: $\text{TTD} = \sum_{v \in \text{VMs}} TD_v$, where $TD_v = T_{\text{int}} + T_{\text{classify}} + T_{\text{migration}} \le 5\text{ min} + T_{\text{migration}}$.

---

## 5. Experimental Configuration & Baselines

- **Proposed Algorithm**: Least-Loaded + 5-min Observation + AE (5 latent) + SVM + MMT Migration + All-or-Nothing Consolidation + 25% CPU-intensive VM limit.
- **Google / Round Robin (RR)**: Round-robin placement with 75% threshold.
- **Modified First Fit (MFFA)**: First fit with 75% threshold.
- **Modified Best Fit (MBFA)**: Best fit with 75% threshold.
- **Sercon**: Greedy consolidation from least-loaded to most-loaded PMs ($Th=75\%, T_{\text{lower}}=35\%$).
- **Configurable Modes**:
  - `dev_mode`: 100 PMs, 1000 VMs for fast execution.
  - `reproduction_mode`: 798 PMs, 7644 VMs matching paper.
