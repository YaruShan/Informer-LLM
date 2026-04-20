# Informer-LLM

Code for our Informer-LLM paper on time series forecasting with Informer and large language models.

## Overview
Informer-LLM is a framework for multivariate time series forecasting that combines Informer-based temporal modeling with large language models.

## Requirements
- Python 3.9+
- PyTorch
- Transformers
- NumPy
- Pandas
- scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
Informer-LLM/
├─ README.md
├─ requirements.txt
├─ THIRD_PARTY_NOTICES.md
├─ config.py
├─ train.py
├─ data/
│  └─ __init__.py
├─ models/
│  ├─ __init__.py
│  ├─ revin.py
│  ├─ patching.py
│  ├─ encoder.py
│  ├─ cpp_llm.py
│  └─ informer_llm.py
├─ utils/
│  ├─ __init__.py
│  ├─ seed.py
│  ├─ metrics.py
│  └─ dataset.py
└─ tests/
   ├─ step1_test_dataset.py
   ├─ step2_test_revin_patch.py
   ├─ step3_test_encoder.py
   ├─ step4_test_cpp_llm.py
   └─ step5_test_full_model.py
```

## Step-by-step Test

Run the following scripts one by one:

```bash
python tests/step1_test_dataset.py
python tests/step2_test_revin_patch.py
python tests/step3_test_encoder.py
python tests/step4_test_cpp_llm.py
python tests/step5_test_full_model.py
```

## Training

Run full training with:

```bash
python train.py
```

## Notes

- This repository provides a clean and runnable baseline implementation of Informer-LLM.
- The implementation includes:
  - RevIN normalization
  - patch embedding for time series
  - transformer encoder for temporal representation learning
  - contextual prompt + frozen LLM backbone
  - output projection for forecasting

## Acknowledgements

Parts of this project were developed with reference to the following open-source repositories:

- Informer: https://github.com/zhouhaoyi/Informer2020
- Time-LLM: https://github.com/KimMeen/Time-LLM

Both projects are licensed under Apache License 2.0.