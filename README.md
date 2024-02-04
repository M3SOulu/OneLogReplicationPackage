# OneLogReplicationPackage

This is a replication package for "OneLog: Towards End-to-end Software Log Anomaly Detection".
To replicated the results follow the steps below:
1. Clone the repository
    ```bash
    git clone https://github.com/M3SOulu/OneLogReplicationPackage.git
    ```
1. Create environment and install dependencies
    ```bash
    conda env create -f environment.yml
    ```
1. Activate environment
    ```bash
    conda activate onelog
    ```
1. Run the expriment with the config:
    ```bash
    python trainer.py fit --config configs/{config}.yaml
    ```