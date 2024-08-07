# language_parser_using_transformer
A langauge parser using Text-To-Text Transfer Transformer (T5) is a pre-trained encoder-decoder model handling all NLP tasks as a unified text-to-text-format where the input and output are always text strings. T5-Small is the checkpoint with 60 million parameters.
# ML Model Training for Phone-based Representation of Words

## Project Overview

This project is an experiment aimed at training a Machine Learning (ML) model to accurately parse new words into their phonetic representations. The model is trained using a curated and verified dataset of words and their corresponding phonetic outputs. Over time, the model learns to generalize and correctly represent new words in a phonetic format.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- Other dependencies listed in `requirements.txt`

### Dataset Preparation

1. **Create Dataset**

   The dataset is prepared using the `create_dataset.py` script. This script shuffles and divides the data into training and validation sets.

   ```bash
   python create_dataset.py --input_data path_to_data --output_dir path_to_output
  ```
2. **Extend Tokenizer

   To handle new UTF-8 tokens (including non-English characters), the tokenizer needs to be extended. This is done using the extend_tokenizer.py script.

  ```bash
python extend_tokenizer.py --tokenizer_path path_to_tokenizer --data_path path_to_extended_tokens
```
3. **Model Training**
Once the dataset is ready and the tokenizer is extended, you can proceed to train the model using a pretrained checkpoint of the T5-small model. This is done using the train.py script.
  ```bash
python train.py --train_data path_to_train_data --val_data path_to_val_data --model_checkpoint t5-small --output_dir path_to_trained_model

```
4. **Inference**
After training, the model can be used for inference on new words. Use the inference_using_checkpoint.py script to perform inference.
```bash
python inference_using_checkpoint.py --model_checkpoint path_to_trained_model --input_data path_to_new_words --output_file path_to_results
```
5. **Directory Structure**
   ```bash
   ├── create_dataset.py
├── extend_tokenizer.py
├── train.py
├── inference_using_checkpoint.py
├── requirements.txt
├── data
│   ├── raw
│   ├── processed
│   └── extended_tokens
├── models
│   └── t5-small
├── output
│   ├── checkpoints
│   └── results
└── README.md
```
