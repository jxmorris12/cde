import pyinstrument
import torch
import transformers
import tqdm

from collate import DocumentQueryCollatorWithPadding
from dataset import load_synthetic_chars_dataset
from trainer import CustomTrainer

def main(num_iters: int, num_workers: int):
    train_dataset, _ = load_synthetic_chars_dataset()
    embedder_tokenizer =  transformers.AutoTokenizer.from_pretrained(
        'bert-base-uncased',
        use_fast=True,
    )

    collator = DocumentQueryCollatorWithPadding(
        tokenizer=embedder_tokenizer,
        padding='longest',
        return_tensors='pt',
        max_length=64,
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    for _ in range(num_iters):
        for _ in tqdm.tqdm(dataloader):
            pass

if __name__ == '__main__':
    profiler = pyinstrument.Profiler()
    profiler.start()

    main(
        num_iters=3,
        num_workers=0,
    )

    profiler.stop()
    print(
        profiler.output_text(
            show_all=True, 
            unicode=False, 
            color=False
        )
    )