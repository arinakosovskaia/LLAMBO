# Enhanced Hyperparameter Search via Bayesian Optimization with Chain of Thoughts Prompting

This repository is a fork of the [original LLAMBO project](https://github.com/tennisonliu/LLAMBO). It explores whether hyperparameter search through Bayesian optimization can be improved using Chain of Thoughts (CoT) prompting. I made a fix for the `rate_limiter`, modifications to collect regrets, updated the code for the new version of OpenAI, and adapted the code for the CoT task.

## 📦 Setup Instructions

### 1. Environment Variables
Before running the code, set up the required environment variables. Add the following lines to your `~/.zshrc` file:

```zsh
BASE_URL=
API_KEY=
```

#### For OpenAI GPT models:
- Set `BASE_URL` to `https://api.openai.com/v1`.
- Obtain your `API_KEY` from [OpenAI API Keys](https://platform.openai.com/api-keys).

#### For Nebius AI Studio:
- Set `BASE_URL` to `https://api.studio.nebius.ai/v1/`.
- Obtain your `API_KEY` from [Nebius AI Studio API Keys](https://studio.nebius.ai/settings/api-keys).

After adding the variables, refresh your shell environment:

```zsh
source ~/.zshrc
```
---

### 2. Setting Up the Conda Environment
Follow these steps to set up the environment:

1. Clone the repository:
```bash
   git clone https://github.com/arinakosovskaia/LLAMBO.git
```

2. Create a Conda environment:
```bash
   conda create -n llambo python=3.8.8
```
3. Install Jupyter:
```bash
   conda install jupyter
```
4. Activate the environment:
```bash
   conda activate llambo
```

5. Set up project-specific environment variables:
   Replace `{project_dir}` with the path to your local directory:
```bash
   export PROJECT_DIR={project_dir}
   conda env config vars set PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
   conda env config vars set PROJECT_DIR=${PROJECT_DIR}
```
6. Reload and activate the environment:
```bash
   conda deactivate
   conda activate llambo
```
7. Install required Python packages:
```bash
   pip install -r requirements.txt
```
### 3. Running Experiments

#### End-to-End Demonstration of the Algorithm
To run an end-to-end demonstration of the algorithm, use the script ./run_prompt_ablation.sh:
```bash
./run_prompt_ablation.sh
```
This script accepts the following parameters:
- ENGINE: The large language model used for generation (e.g., OpenAI GPT).
- prompting: The type of prompting to use:
  - "cot" for Chain of Thoughts (CoT) few-shot prompting.
  - "few_shot" for standard few-shot prompting.
- max_reasoning_tokens: The maximum number of tokens the model can use for reasoning.

#### Evaluating LLM as a Discriminative Surrogate Model
To evaluate the large language model (LLM) as a discriminative surrogate model with different numbers of observations and types of prompting, use the script ./run_evaluate_dis_sm.sh:
```bash
./run_evaluate_dis_sm.sh
```
You can specify the ENGINE parameter to select the LLM for evaluation.

#### Evaluating LLM as a Candidate Sampler
To evaluate the LLM as a candidate sampler, use the script ./run_evaluate_sampling.sh:
```bash
./run_evaluate_sampling.sh
```

---

### 4. Generating Plots

Detailed descriptions and instructions for generating plots can be found in the `make_plots` folder. 
```bash
cd make_plots
```