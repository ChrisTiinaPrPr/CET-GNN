# Dataset files

Download the original datasets from their official repositories.

- **NF-UNSW-NB15-v2:** https://doi.org/10.48610/ffbb0c1
- **NF-CSE-CIC-IDS2018-v3:** https://doi.org/10.48610/ece9b83
- **GridStratLLM:** [RADAR4KIT download](https://radar.kit.edu/radar/en/dataset/bx5337kcykte438h) · [DOI record](https://doi.org/10.35097/bx5337kcykte438h)

For GridStratLLM, extract `merged_datasets.duckdb` here and follow the [preparation, training, and inference commands](../README.md#gridstratllm-training-and-inference).

The CIC V3 download may use the filename `NF-CICIDS2018-v3.csv`.
Alternatively, use `train.py --data /path/to/dataset.csv`.

The repository excludes downloaded data and generated arrays. See the root README for the data protocols and training commands.
