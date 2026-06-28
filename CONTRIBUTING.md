# Contributing to UATC

Thank you for your interest in contributing to **UATC (Universal Adaptive Training Controller)**! We welcome contributions from developers, machine learning engineers, researchers, and system architects to help improve and expand this closed-loop resource controller.

To maintain codebase stability, security, and academic rigor, please adhere to the guidelines and workflows outlined below.

---

## Code of Conduct

By participating in this project, you agree to maintain a professional, collaborative, and respectful environment. Please ensure all communications, issue reports, and pull requests remain constructive and technically objective.

---

## How Can I Contribute?

### 1. Reporting Bugs & Issues

If you encounter a CUDA Out-Of-Memory (OOM) crash that UATC failed to intercept, a numerical instability (NaN/Inf), or a runtime error:

1. Navigate to the [GitHub Issues](https://github.com/sajjaddoda72-design/UATC/issues) page.
2. Open a new issue and select the **Bug Report** template.
3. Provide a minimal, reproducible code snippet (or a Google Colab link) alongside your specific hardware configuration (GPU model, total VRAM, and QLoRA/PEFT hyperparameters).
4. Include the exact error trace and the console logs printed prior to the failure.

### 2. Suggesting Enhancements

If you have ideas for expanding UATC (e.g., adding advanced state estimators like Particle Filters, supporting multi-node cluster configurations, or adding new training paradigm presets):

1. Open a new issue in [GitHub Issues](https://github.com/sajjaddoda72-design/UATC/issues).
2. Describe the feature, its theoretical utility, and how it aligns with the closed-loop control philosophy of UATC.

---

## Development & Contribution Workflow

To modify the codebase, fix bugs, or add features, please follow this step-by-step distributed workflow.

### Step 1: Fork and Clone the Repository

1. Fork the official repository to your own GitHub account:
   ```
   https://github.com/sajjaddoda72-design/UATC.git
   ```
2. Clone your fork locally to your workstation or cloud environment:
   ```bash
   git clone https://github.com/<your-username>/UATC.git
   cd UATC
   ```

### Step 2: Set Up Your Development Environment

UATC uses modern Python packaging standards. We recommend setting up a clean virtual environment and installing dependencies using the local standard packaging directive:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -e .
```

> **Note:** Installing with `-e .` reads dependencies directly from `pyproject.toml` and installs UATC as an editable module.

### Step 3: Implement Your Changes

- All controller modifications and new system behaviors must be implemented inside `UATC.py`.
- Ensure your code conforms strictly to **PEP 8** style guidelines and includes clear, professional docstrings written in English.

### Step 4: Run the Automated Test Suite (Crucial Step)

Before submitting any changes, you must verify that your modifications have not broken the core mathematical and control logic. UATC maintains a robust local unit testing suite inside the `tests/` directory.

Run the tests locally using Python's standard discovery:

```bash
python -m unittest discover -s tests
```

Your code will only be accepted if all tests pass successfully. If you introduce new parameters or subsystems, you are expected to write corresponding test cases inside `tests/test_controller.py`.

### Step 5: Submit a Pull Request (PR)

1. Commit your changes with a clear, descriptive commit message:
   ```bash
   git commit -m "Fix PID derivative filter wind-up in PEFT mode"
   ```
2. Push the changes to your remote GitHub fork:
   ```bash
   git push origin feature/your-feature-name
   ```
3. Go to the original `sajjaddoda72-design/UATC` repository on GitHub and click **Compare & pull request**.
4. Write a concise description explaining what your PR changes, why it is necessary, and confirm that all unit tests have passed locally.

---

## Pull Request Review & Merging Process

Once submitted:

1. **Automated CI Checks:** Our GitHub Actions Continuous Integration (CI) workflow will automatically spin up a runner, install dependencies, and execute the full unit test suite on your branch.
2. **Code Review:** The maintainers will review the code for performance overhead, stability impact, and mathematical correctness.
3. **Approval:** Upon successful code review and green CI checks, your pull request will be merged into the main stable branch of UATC.

---

Thank you for helping us make deep learning training more stable and resource-efficient! 🚀