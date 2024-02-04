# OneLog: Towards End-to-End Software Log Anomaly Detection

## Abstract

With the rapid expansion of online services, IoT devices, and DevOps-oriented software development practices, the need for efficient software log anomaly detection has grown significantly. Traditional approaches to this challenge typically rely on a four-staged architecture, encompassing a Preprocessor, Parser, Vectorizer, and Classifier. This paper introduces **OneLog**, a novel method that employs a single Deep Neural Network (DNN) to streamline the process, replacing the need for multiple distinct components.

**OneLog** leverages the power of Convolutional Neural Networks (CNN) at the character level, incorporating digits, numbers, and punctuation—elements often omitted in previous studies—alongside the main body of natural language text. Our evaluation of this approach spans six message- and sequence-based datasets: HDFS, Hadoop, BGL, Thunderbird, Spirit, and Liberty, exploring its application in single-, multi-, and cross-project setups.

Our findings reveal that **OneLog** not only achieves state-of-the-art performance across our datasets but also demonstrates an impressive capacity to generalize between datasets when trained on multi-project data. This feature is particularly advantageous in scenarios where individual projects may suffer from limited training data. Additionally, our experiments indicate the feasibility of cross-project anomaly detection, with successful trials involving a single project pair (Liberty and Spirit). Further analysis of **OneLog**'s internal mechanisms uncovers its multifaceted approach to anomaly detection and its ability to autonomously learn parsing rules that have been manually validated for log messages.

We conclude that character-based CNNs represent a promising avenue for end-to-end learning in the field of log anomaly detection, offering notable advantages in performance and generalization across multiple datasets. In line with our commitment to the research community, we will make our scripts publicly available following the acceptance of this paper.

## Results
| **Method** | **HDFS** | **Hadoop** | **BGL** | **Thunderbird** | **Spirit** | **Liberty** |
|------------|---------:|-----------:|--------:|----------------:|-----------:|------------:|
| **OneLog** | **0.99** | **0.97**   | **0.99**| **0.99**        | **0.99**   | **0.99**    |
| LogBERT    | 0.82     | N.A        | 0.91    | 0.97            | N.A        | N.A         |
| NeuralLog  | 0.98     | N.A        | 0.98    | 0.96            | 0.97       | N.A         |
| Logsy      | N.A      | N.A        | 0.65    | 0.99            | 0.99       | N.A         |
| LogRobust  | 0.99     | 0.90       | 0.75    | 0.68            | 0.95       | N.A         |
| LogAnomaly | 0.94     | N.A        | 0.88    | 0.84            | 0.95       | N.A         |
| DeepLog    | 0.95     | N.A        | 0.86    | 0.93            | N.A        | N.A         |
| SiaLog     | 0.99     | 0.94       | 0.99    | N.A             | N.A        | N.A         |
| CNNLog     | 0.98     | 0.92       | 0.95    | N.A             | N.A        | N.A         |
| Auto-LSTM  | N.A      | N.A        | 0.95    | 0.99            | N.A        | N.A         |
## Results Replication
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

## Cite this work
```bibtex
filled uppon publish
```