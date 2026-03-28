"""
generate.py — Run inference with a trained EEG-LLM model.

Loads a saved projection + LoRA adapter and generates text descriptions
from EEG input samples.

Usage:
    python generate.py \
        --foundation_dir pth/CSBrain.pth \
        --projection_dir pth_downtasks/eeg_llm_bcic/best_projection.pth \
        --lora_dir pth_downtasks/eeg_llm_bcic/best_lora \
        --datasets_dir data/BCICIV2a/processed_lmdb \
        --downstream_dataset BCICIV2a \
        --num_samples 5
"""

import argparse
import random
import numpy as np
import torch

from datasets.bciciv2a_llm_dataset import LoadDataset as BCICDataset, BCIC_LABEL_MAP
from models.eeg_llm import EEGLanguageModel



def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(params) -> EEGLanguageModel:
    """Build model and load saved weights."""
    from peft import PeftModel

    model = EEGLanguageModel(params)
    device = next(model.parameters()).device

    # Load projection + token reducer weights
    state = torch.load(params.projection_dir, map_location=device)
    if 'projection' in state:
        model.eeg_projection.load_state_dict(state['projection'])
        if 'token_reducer' in state:
            model.token_reducer.load_state_dict(state['token_reducer'])
    else:
        model.eeg_projection.load_state_dict(state)
    print(f"Loaded projection from {params.projection_dir}")

    # Load LoRA adapter
    model.llm = PeftModel.from_pretrained(model.llm, params.lora_dir)
    print(f"Loaded LoRA adapter from {params.lora_dir}")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--downstream_dataset', type=str, default='BCICIV2a')
    parser.add_argument('--datasets_dir', type=str, required=True)
    parser.add_argument('--num_of_classes', type=int, default=4)
    parser.add_argument('--model_dir', type=str, default='pth_downtasks/eeg_llm_bcic')
    parser.add_argument('--foundation_dir', type=str, default='pth/CSBrain.pth')
    parser.add_argument('--projection_dir', type=str,
                        default='pth_downtasks/eeg_llm_bcic/best_projection.pth')
    parser.add_argument('--lora_dir', type=str,
                        default='pth_downtasks/eeg_llm_bcic/best_lora')
    parser.add_argument('--llm_model_name', type=str,
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--llm_dim', type=int, default=2048)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--max_target_len', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--use_pretrained_weights', action='store_true', default=True)
    parser.add_argument('--temporal_pool_stride', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of test samples to generate text for')
    params = parser.parse_args()

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)

    print("Loading model...")
    model = load_model(params)

    print("Loading test data...")
    load_dataset = BCICDataset(params, model.tokenizer)
    data_loaders = load_dataset.get_data_loader()
    test_loader = data_loaders['test']

    BCIC_LABEL_TEXTS = {0: "left hand", 1: "right hand", 2: "feet", 3: "tongue"}
    device = next(model.parameters()).device

    print(f"\nGenerating text for {params.num_samples} test samples:\n")
    count = 0
    for batch in test_loader:
        if count >= params.num_samples:
            break

        generated_texts = model.generate(
            eeg_data=batch['eeg_data'].to(device),
            prompt_ids=batch['prompt_ids'].to(device),
            prompt_mask=batch['prompt_mask'].to(device),
            max_new_tokens=64,
        )

        for i, (text, true_label) in enumerate(zip(generated_texts, batch['label_ids'].tolist())):
            if count >= params.num_samples:
                break
            print(f"Sample {count + 1}:")
            print(f"  True class : {true_label} — {BCIC_LABEL_TEXTS.get(true_label, str(true_label))}")
            print(f"  Generated  : {text}")
            print()
            count += 1


if __name__ == '__main__':
    main()
