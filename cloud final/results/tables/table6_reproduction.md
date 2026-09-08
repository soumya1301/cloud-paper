# Table 6: Experimental Comparison of VM Placement Algorithms

| Algorithm          | AUR    |   # active PMs |   # Idle PMs |   TEC (wph) |   PDM |   SLATAH |   # migrated VMs |   Migration Cost |   Overloaded PMs |   Underloaded | SLAv   |
|:-------------------|:-------|---------------:|-------------:|------------:|------:|---------:|-----------------:|-----------------:|-----------------:|--------------:|:-------|
| Google             | 52.86% |            798 |            0 |    102932   |     0 |  0.04505 |                0 |                0 |                0 |             0 | No     |
| MBFA               | 45.02% |            798 |            0 |     97577.7 |     0 |  0       |                0 |                0 |                0 |             0 | No     |
| MFFA               | 45.12% |            798 |            0 |     97589   |     0 |  0       |                0 |                0 |                0 |             0 | No     |
| RRA                | 52.27% |            798 |            0 |    102386   |     0 |  0.05805 |                0 |                0 |                0 |             0 | No     |
| Sercon             | 45.02% |            798 |            0 |     97577.7 |     0 |  0       |                0 |                0 |                0 |             0 | No     |
| Proposed Algorithm | 45.34% |            798 |            0 |     97612.4 |     0 |  0       |                0 |                0 |              757 |             0 | Yes    |