# AI2 — Model Poisoning Tool

Data poisoning simulation and backdoor injection toolkit for ML security research.

## Overview

This project demonstrates how machine learning models can be compromised through poisoned training data:
- **Label flipping**: Corrupt training labels to degrade model performance
- **Random noise injection**: Add noisy samples to corrupt decision boundaries
- **Backdoor injection**: Embed trigger patterns that activate malicious behavior
- **Training data manipulation**: Undersampling, outlier injection, duplication

## Features

- **Data Poisoning Attacks**: Multiple poisoning strategies (label flip, noise, feature manipulation)
- **Backdoor Injection**: Embed triggers that cause misclassification to a target class
- **Training Data Manipulation**: Undersample, duplicate, and add outliers
- **Clean Model Training**: Simple logistic regression for demonstration
- **Impact Assessment**: Measure accuracy degradation from each attack type

## Installation

```bash
pip install numpy
```

## Usage

```python
from model_poison import CleanModel, DataPoisoner, BackdoorInjector

# Poison training data
poisoner = DataPoisoner()
x_poisoned, y_poisoned, indices = poisoner.label_flip(x_train, y_train, poison_rate=0.3)

# Inject backdoor
injector = BackdoorInjector(trigger_size=3)
x_bd, y_bd, trigger, bd_indices = injector.inject(
    x_train, y_train, target_label=0, poison_rate=0.1)
```

### Running the Demo

```bash
python3 model_poison.py
```

## Example Output

```
============================================================
  AI2 — Model Poisoning Tool Demo
============================================================

Clean model accuracy: 100.00%

--- Label Flip Poisoning ---
  Poisoned accuracy: 76.00% (was 100.00%)
  Poisoned samples: 150

--- Backdoor Injection ---
  Clean accuracy: 100.00%
  Backdoor success: 100.00%
  Trigger pattern: [3, 7, 12]

--- Summary ---
  Clean:             100.00%
  Label Flip:        76.00%
  Random Noise:      97.00%
  Backdoor:          100.00%
  Undersampling:     74.00%
  Outlier Injection: 95.00%
```

## Attack Types

### Label Flip
Randomly changes a portion of training labels to incorrect classes.

### Random Noise
Adds Gaussian noise to a subset of training features.

### Feature Poison
Manipulates specific feature values in training samples.

### Backdoor Injection
Adds a trigger pattern to clean samples and relabels them to a target class.

## Legal Disclaimer

**IMPORTANT: Read before use.**

This project is provided for **educational and authorized security testing purposes only**.

### Authorization Requirements
- You MUST have explicit written permission before testing data poisoning on any system
- Unauthorized manipulation of training data is illegal and unethical
- This tool should ONLY be used on models you own or have written authorization to test

### Legal Framework
- **Computer Fraud and Abuse Act (CFAA)**: Unauthorized access to computer systems is a federal crime
- **GDPR/CCPA**: Manipulating training data may violate data protection regulations
- **State Laws**: Many states have additional computer crime statutes
- **AI Safety Regulations**: Emerging laws specifically address AI model integrity

### Acceptable Use
- Testing robustness of your own ML models
- Authorized red team assessments with written scope
- Academic research in controlled lab environments
- Security education and training

### Prohibited Use
- Poisoning models you do not own without authorization
- Injecting backdoors into production systems
- Any activity that violates applicable laws or regulations
- Commercial use without proper licensing

### No Warranty
This software is provided "AS IS" without warranty of any kind. The author is not responsible for any misuse or damage caused by this software.

### Responsible Disclosure
If you discover vulnerabilities using this tool, follow responsible disclosure practices:
1. Report to the vendor/owner privately
2. Allow reasonable time for remediation
3. Do not exploit beyond proof of concept

## License

MIT
