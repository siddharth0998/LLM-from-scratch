import torch
from torch.utils.data import Dataset

from .chat_format import encode_conversation, instruction_entry_to_messages

IGNORE_INDEX = -100
PAD_TOKEN_ID = 50256


class SFTDataset(Dataset):

    def __init__(self, data, tokenizer, max_length=1024):
        self.examples = []
        for entry in data:
            messages = (
                entry if isinstance(entry, list)
                else instruction_entry_to_messages(entry)
            )
            token_ids, assistant_mask = encode_conversation(messages, tokenizer)

            input_ids = token_ids[:-1]
            target_ids = token_ids[1:]
            target_mask = assistant_mask[1:]

            targets = [
                tid if keep else IGNORE_INDEX
                for tid, keep in zip(target_ids, target_mask)
            ]

            input_ids = input_ids[:max_length]
            targets = targets[:max_length]

            if not any(t != IGNORE_INDEX for t in targets):
                continue

            self.examples.append((input_ids, targets))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

def sft_collate_fn(batch, pad_token_id=PAD_TOKEN_ID, ignore_index=IGNORE_INDEX,
                   allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(inp) for inp, _ in batch)
    if allowed_max_length is not None:
        batch_max_length = min(batch_max_length, allowed_max_length)

    inputs_lst, targets_lst = [], []
    for input_ids, target_ids in batch:
        input_ids = input_ids[:batch_max_length]
        target_ids = target_ids[:batch_max_length]

        pad = batch_max_length - len(input_ids)
        inputs_lst.append(input_ids + [pad_token_id] * pad)
        targets_lst.append(target_ids + [ignore_index] * pad)

    inputs_tensor = torch.tensor(inputs_lst, dtype=torch.long).to(device)
    targets_tensor = torch.tensor(targets_lst, dtype=torch.long).to(device)
    return inputs_tensor, targets_tensor

class InstructionDataset(Dataset):
    """Deprecated: use SFTDataset. Does not mask the prompt."""

    def __init__(self, data, tokenizer):
        from .chat_format import format_input
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch, pad_token_id=PAD_TOKEN_ID, ignore_index=IGNORE_INDEX,
                      allowed_max_length=None, device="cpu"):
    """Deprecated collate for InstructionDataset (masks padding only)."""
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
