# ML-Based Virtual Machine Placement in Data Centres

A faithful Python reproduction of the research paper:
> **"A Machine Learning Technique for Optimizing Virtual Machine Placement in Data-Centres"**  
> *Abdullah Alelyani, Amitava Datta, Ghulam Mubashar Hassan*  
> **Journal of Cloud Computing**, Vol. 15, Article 37 (2026)  
> DOI: [10.1186/s13677-026-00851-3](https://doi.org/10.1186/s13677-026-00851-3)

---

## 1. Project Overview

Cloud data centers face major challenges balancing resource utilization, energy consumption, and Service Level Agreement (SLA) violations. This paper proposes a machine learning-driven Virtual Machine (VM) placement and consolidation algorithm that:
1. Observes newly arriving VMs for a **5-minute observation window** to collect resource usage patterns.
2. Extracts **16 VM resource features** (CPU, instruction-level efficiency CPI/MAI, memory, and duration metrics).
3. Clusters VMs into **3 distinct classes** using **K-Means** ($K=3$).
4. Compresses the 16 features into **5 latent features** using a **PyTorch Autoencoder** ($16 \to 12 \to 5 \to 12 \to 16$).
5. Classifies newly arriving VMs in real time using a **Support Vector Machine (SVM)** with an RBF kernel.
6. Deploys VMs to the **least-loaded Physical Machine (PM)** considering both CPU and memory demand, enforces an upper utilization threshold ($Th = 75\%$), and constrains CPU-intensive VMs to **$\le 25\%$ per PM**.
7. Executes dynamic consolidation with **Minimum Migration Time (MMT)** overloaded PM relief and **All-or-Nothing** underloaded PM consolidation (putting empty PMs into a **10 W sleep mode**).

---

## 2. Dataset: Google Cluster Workload Trace

- **Official Google Trace Repository**: [https://github.com/google/cluster-data](https://github.com/google/cluster-data)
- **Raw Data Characteristics**: The raw Google dataset (`clusterdata-2011-2`) contains over 2 TB of trace data covering jobs and machine events over 29 days.
- **Subset Extraction**: Rather than downloading the entire 2 TB trace, this implementation provides an automated extractor (`data/download_subset.py`) that extracts the exact dataset subsets used in the paper:
  - **ML Dataset**: 2,293 VMs (70% Train, 10% Validation, 20% Test) matching the paper's cluster distribution (Cluster 1: 926, Cluster 2: 407, Cluster 3: 922).
  - **Scheduling Dataset**: 7,644 VMs deployed across 798 heterogeneous PMs (HP ProLiant G5 servers).

### The 16 Extracted VM Features

| Feature | Description | Field / Source |
|:---|:---|:---|
| `cpu_max` | Maximum CPU utilization | $\max(\text{CPU usage})$ in 5-min window |
| `cpu_min` | Minimum CPU utilization | $\min(\text{CPU usage})$ in 5-min window |
| `cpu_avg` | Mean CPU utilization | $\text{mean}(\text{CPU usage})$ in 5-min window |
| `cpu_std` | Standard deviation of CPU usage | $\text{std}(\text{CPU usage})$ in 5-min window |
| `cpu_p95` | 95th percentile CPU usage | Peak demand metric |
| `cpi` | Cycles Per Instruction | Google `task_usage.cpi` |
| `mai` | Memory Accesses Per Instruction | Google `task_usage.mai` |
| `mem_max` | Maximum memory utilization | $\max(\text{Memory usage})$ in window |
| `mem_min` | Minimum memory utilization | $\min(\text{Memory usage})$ in window |
| `mem_avg` | Average memory utilization | $\text{mean}(\text{Memory usage})$ in window |
| `mem_assigned` | Assigned memory | Google `task_usage.assigned_memory` |
| `mem_page_cache` | Page cache memory | Google `task_usage.page_cache_memory` |
| `mem_canonical` | Canonical memory usage | Google `task_usage.canonical_memory` |
| `start_time` | Start timestamp | Job arrival timestamp |
| `end_time` | End timestamp | Job end timestamp |
| `duration` | Total runtime window | $\text{end\_time} - \text{start\_time}$ |

---

## 3. Machine Learning Architecture

```
Raw 16 Features -> Normalized [0, 1] -> K-Means (K=3) -> Cluster Labels
                                      \
                                       -> PyTorch Autoencoder (16 -> 12 -> 5 -> 12 -> 16)
                                            \
                                             -> 5 Latent Features -> SVM (RBF) -> VM Class Prediction
```

- **K-Means Clustering ($K=3$)**:
  - **Class 1 (Balanced / Low CPU)**: Mean CPU $\approx 0.028$, Memory $\approx 0.05$.
  - **Class 2 (CPU-Intensive)**: Mean CPU $\approx 0.170$ (10x higher), negligible memory.
  - **Class 3 (Moderate CPU / Low Memory)**: Mean CPU $\approx 0.0542$, Memory $\approx 0.0089$.
- **PyTorch Autoencoder**:
  - Configuration: $16 \to 12 \to 5 \to 12 \to 16$.
  - Hidden activation: ReLU; Output activation: Sigmoid.
  - Optimizer: Adam, Loss: MSE, Epochs: 500, Batch Size: 64.
  - Compresses 16 features to 5 latent features with $>98\%$ reconstruction accuracy.
- **SVM Classifier**:
  - RBF kernel with 5-fold cross-validation on $C$ and $\gamma$.
  - Evaluates Precision ($\approx 79\%$), Recall/Sensitivity ($\approx 80\%$), and F1-Score ($\approx 0.77$).

---

## 4. Datacenter Simulation & Mathematical Equations

### Physical Machine Model (HP ProLiant G5 Server)
- 798 PMs with 1.0 GCU CPU, 1.0 GCU Memory, 4 Gbps Network Bandwidth ($500\text{ MB/s}$).
- SPECpower Benchmark Power Lookup Table:
  | CPU Load (%) | Sleep | 0% | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | **Power (W)** | 10.0 | 93.7 | 97.0 | 101.0 | 105.0 | 110.0 | 116.0 | 121.0 | 125.0 | 129.0 | 133.0 | 135.0 |

### Mathematical Formulas
1. **Initial Priority Queue**:
   $$\text{Priority}_{\text{initial}} = \text{sort}(\text{VMs}, \{A_i \uparrow, -P_i \downarrow, -R_i \downarrow\})$$
2. **Resource Utilization Ratio ($R^d_p$)**:
   $$R^d_p = \frac{U^d_p}{C^d_p} \quad (d \in \{\text{CPU}, \text{Memory}\})$$
3. **VM Migration Time ($T$)**:
   $$T = \frac{VM_{\text{memory}}}{B_w}$$
4. **CPU Degradation during Migration ($TD_{\text{VM}_j}$)**:
   $$TD_{\text{VM}_j} = 0.1 \times \int_{t_{\text{start}}}^{T} U^{\text{cpu}} dt = 0.1 \times U^{\text{cpu}} \times T$$
5. **SLATAH (SLA Violation Time per Active Host)**:
   $$\text{SLATAH} = \frac{1}{N} \sum_{i=1}^N \frac{T^i_{\text{un}}}{T_{\text{an}}}$$
6. **PDM (Performance Degradation due to Migrations)**:
   $$\text{PDM} = \frac{1}{M} \sum_{j=1}^M \frac{C^j_{\text{dm}}}{C^j_{\text{rm}}}$$
7. **Overall SLA Violations ($SLAv$)**:
   $$\text{SLAv} = \text{SLATAH} \times \text{PDM}$$
8. **Total Time Delay (TTD)**:
   $$\text{TTD} = \sum_{v \in \text{VMs}} TD_v \quad \text{where } TD_v = T_{\text{int}} + T_{\text{classify}} + T_{\text{migration}} \le 5\text{ min} + T_{\text{migration}}$$

---

## 5. Repository Structure

```
vm-placement-paper/
├── config/
│   └── config.yaml               # Central experiment configuration
├── data/
│   ├── download_subset.py        # Automated trace downloader & subset generator
│   ├── raw/                      # Raw trace tables
│   └── processed/                # Scaled feature matrices & split datasets
├── models/
│   ├── kmeans.pkl                # Trained K-Means (K=3) model
│   ├── autoencoder.pth           # Trained PyTorch Autoencoder
│   └── svm.pkl                   # Trained SVM classifier
├── src/
│   ├── data_preprocessing.py     # Clean raw trace, filter invalid records, windowing
│   ├── feature_extraction.py     # 16-feature computation from 5-min trace window
│   ├── clustering.py             # K-Means (K=3) with cluster profiling
│   ├── autoencoder.py            # PyTorch Autoencoder & latent representation extractor
│   ├── svm_classifier.py         # SVM trainer, cross-validator, and predictor
│   ├── physical_machine.py       # PhysicalMachine class with SPECpower profile
│   ├── virtual_machine.py        # VirtualMachine class with type, requirements, stats
│   ├── scheduler.py              # Priority-based initial placement + ML integration
│   ├── monitoring.py             # Continuous PM state monitoring (Over/Under/Normal)
│   ├── migration.py              # MMT live migration engine, bandwidth, downtime
│   ├── consolidation.py          # All-or-nothing consolidation & sleep power manager
│   ├── energy.py                 # SPECpower HP G5/G4 energy calculations
│   ├── sla.py                    # SLATAH, PDM, and SLAv metrics
│   └── evaluation.py             # AUR, active PMs, migration cost, TTD aggregator
├── experiments/
│   ├── train_ml.py               # Complete ML training pipeline
│   ├── run_scheduler.py          # Proposed scheduler experiment on 7,644 VMs
│   └── run_comparison.py         # Table 6 comparative benchmark evaluation
├── notebooks/
│   ├── 01_dataset_analysis.ipynb # Dataset exploratory data analysis
│   ├── 02_kmeans.ipynb           # K-Means clustering and cluster profiling
│   ├── 03_autoencoder.ipynb      # Autoencoder training & latent visualization
│   ├── 04_svm.ipynb              # SVM training, confusion matrix, ROC
│   └── 05_results.ipynb          # End-to-end benchmark comparison & plots
├── results/
│   ├── figures/                  # Publication figures (Fig 1, 5, 6, 7, 8, 9, comparison)
│   ├── tables/                   # Table 5 & Table 6 in CSV and Markdown
│   └── logs/                     # Experiment run logs
├── requirements.txt              # Dependencies
├── PAPER_REPRODUCTION_NOTES.md   # Paper vs Implementation comparison table
├── IMPLEMENTATION_PLAN.md        # Technical implementation plan
├── main.py                       # Single CLI entrypoint
└── README.md                     # Project documentation
```

---

## 6. How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
```bash
python main.py all
```

### 3. Run Individual Steps
- **Extract / Download Google Trace Subset**:
  ```bash
  python main.py download
  ```
- **Preprocess Data & Extract 16 Features**:
  ```bash
  python main.py preprocess
  ```
- **Train the ML Pipeline (K-Means $\to$ Autoencoder $\to$ SVM)**:
  ```bash
  python experiments/train_ml.py
  ```
- **Run Proposed Cloud Scheduler Simulation**:
  ```bash
  python experiments/run_scheduler.py
  ```
- **Run Comparative Benchmark Evaluation (Table 6 Reproduction)**:
  ```bash
  python experiments/run_comparison.py
  ```
- **Fast Development Mode (100 PMs, 1000 VMs)**:
  ```bash
  python main.py all --dev
  ```

---

## 7. Comparative Benchmark Results (Table 6)

| Algorithm | AUR | # active PMs | # Idle PMs | TEC (wph) | PDM | SLATAH | # migrated VMs | Migration Cost | Overloaded PMs | Underloaded | SLAv |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Google [27]** | 32.00% | 798 | 0 | 83,313.88 | 0.00000 | 0.00270 | 0 | 0.000000 | 61 | 549 | No |
| **MBFA [24]** | 50.66% | 567 | 231 | 67,351.00 | 0.00000 | 0.00000 | 0 | 0.000000 | 0 | 32 | No |
| **MFFA [24]** | 51.89% | 588 | 210 | 69,335.70 | 0.00000 | 0.00000 | 0 | 0.000000 | 0 | 35 | No |
| **RRA [35]** | 52.88% | 567 | 231 | 67,474.00 | 0.00000 | 0.00000 | 0 | 0.000000 | 0 | 13 | No |
| **Sercon [34]** | 31.22% | 771 | 27 | 80,740.00 | 2.70000 | 0.00210 | 498 | 0.005300 | 60 | 477 | No |
| **Proposed Algorithm** | **62.43%** | **293** | **505** | **40,143.00** | **0.00000** | **0.00000** | **3** | **0.000032** | **0** | **2** | **Yes** |
