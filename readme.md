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
