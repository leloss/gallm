# Paper Research Code and Datasets

This repository contains the research code for the paper "An LLM-Based Genetic Algorithm for Prompt Engineering," by Leandro Loss and Pratikkumar Dhuvad.

## Requirements

To ensure a smooth setup, refer to the `requirements.txt` file for all necessary dependencies.

- **Python Version**: 3.9.16

## Setup Instructions

1. **Configure API Credentials**:
   - Update the `.env` file with your endpoint URLs and API credentials. This codebase utilizes the Azure OpenAI API for experiments.

2. **Model Terminology**:
   - For consistency with the paper, the code follows a structured naming convention for different models. The variable `llm_type` distinguishes between small and large models, categorized as GPT, Mistral, and LLaMA. The specific mappings are as follows:

     - **GPT Models:**
       - `LLM_GPT_SMALL`: GPT-4o-mini
       - `LLM_GPT_LARGE`: GPT-4o
     
     - **Mistral Models:**
       - `LLM_MISTRAL_SMALL`: Mistral NeMo
       - `LLM_MISTRAL_LARGE`: Mistral Large 2
     
     - **LLaMA Models:**
       - `LLM_LLAMA_SMALL`: LLaMA 3.2 11B Instruct
       - `LLM_LLAMA_LARGE`: LLaMA 3.3 70B

3. **Cost Estimation**:
   - Update the cost values in the code to obtain precise cost estimations for your runs.

## Debugging Instructions

To facilitate troubleshooting, a `debug` flag is available at the beginning of the script. If errors occur during execution, set `debug = False` to prevent errors from being suppressed by parallel processing.

## Running the Code

The script requires the following command-line arguments:

- `_dataset` - The dataset file to be used.
- `_nexamples` - The number of examples to process.
- `llm_type` - The type of LLM model (e.g., GPT, Mistral, LLaMA).
- `llm_size` - The size of the model (small or large).

### Dataset Format

Ensure that dataset files are located in the same directory as the script. The first and second columns of the dataset files are utilized for processing.

#### Available Dataset Files:
- `bulgarian-qa.csv`
- `financial_math.csv`
- `science-qa-parsed.csv` # Removed from Supplementary Material due to file size limit. Refer to paper's references for source url.
- `topics.csv`

### Example Usage

Run the script using the following command:

```sh
python3.8 gallm-fit.py <dataset> <number_examples> <model_type> <model_size>

**Note:** The number of examples can significantly affect the input token size, so be cautious when choosing the number of examples for a given LLM to avoid exceeding token limits.

```
#### Example Commands:
```sh
python3.9 gallm-fit.py bulgarian-qa.csv 200 mistral small
python3.9 gallm-fit.py bulgarian-qa.csv 200 gpt large
python3.9 gallm-fit.py topics.csv 40 llama small
python3.9 gallm-fit.py topics.csv 200 gpt large
python3.9 gallm-fit.py truthful-qa-parsed.csv 50 mistral small
```

## Citations

If you use GALLM in your research or project, please cite:

```bibtex
@inproceedings{loss2025cec,
  author = {Loss, Leandro A. and Dhuvad, Pratikkumar},
  title = {From Manual to Automated Prompt Engineering: Evolving LLM Prompts with Genetic Algorithms},
  year = {2025},
  publisher = {IEEE Congress on Evolutionary Computation (CEC'25)},
  url = {https://github.com/leloss/gallm/}
}

@inproceedings{loss2025gecco,
  author = {Loss, Leandro A. and Dhuvad, Pratikkumar},
  title = {An LLM-Based Genetic Algorithm for Prompt Engineering},
  year = {2025},
  publisher = {ACM Genetic and Evolutionary Computation Conference (GECCO'25)},
  url = {https://github.com/leloss/gallm/}
}

@software{loss2025gecco,
  author = {Loss, Leandro A. and Dhuvad, Pratikkumar},
  title = {GALLM: An LLM-Based Genetic Algorithm for Prompt Engineering},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/leloss/gallm/}
}
```
