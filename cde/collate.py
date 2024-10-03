from typing import Any, Dict, List, Optional, Union

import collections
import functools
import os

import torch
import torch.multiprocessing
import transformers


torch.multiprocessing.set_sharing_strategy('file_system')


is_doc = lambda s: s.startswith('document_')
is_hn_doc = lambda s: s.startswith('negative_document_')
is_query = lambda s: s.startswith('query_')
is_dataset_doc = lambda s: s.startswith('dataset_')


def pad_tensor_to_length(the_list: List[int], length: int = 0, value: int = 0) -> List[int]:
    num_pads = length - len(the_list)
    if num_pads == 0:
        return torch.tensor(the_list, dtype=torch.long)
    else:
        return torch.tensor(the_list + [value] * num_pads, dtype=torch.long)


def cut_padding(batch: Dict[str, torch.Tensor], pad_token: int, prefix: str = '') -> Dict[str, torch.Tensor]:
    """Truncates doc if some stuff is all padding at the end."""
    assert f'{prefix}input_ids' in batch
    assert f'{prefix}attention_mask' in batch
    if not (
        isinstance(batch[f'{prefix}attention_mask'], torch.Tensor) and isinstance(batch[f'{prefix}input_ids'], torch.Tensor)): 
        return batch
    # 
    b, s = batch[f'{prefix}input_ids'].shape
    # 
    all_padding = (batch[f'{prefix}input_ids'] == pad_token).all(dim=0)
    if all_padding.sum() == 0:
        return batch
    # 
    padding_start = all_padding.int().argmax()
    batch[f'{prefix}input_ids'] = batch[f'{prefix}input_ids'][:, :padding_start]
    batch[f'{prefix}attention_mask'] = batch[f'{prefix}attention_mask'][:, :padding_start]
    return batch


class TokenizedCollator(transformers.DataCollatorWithPadding):
    tokenizer: transformers.PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    # TODO: Fix to use separate tokenizers

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        os.environ['TOKENIZERS_PARALLELISM'] = '0'

        out_ex = collections.defaultdict(list)
        for ex in features:
            for col in ex:
                out_ex[col].append(ex[col])

        extra_keys = []
        for k, v in out_ex.items():
            if isinstance(v, list) and isinstance(v[0], int):
                out_ex[k] = torch.tensor(v)
            if isinstance(v, list) and isinstance(v[0], list):
                if (len(v[0]) > 0) and isinstance(v[0][0], int):
                    out_ex[k] = torch.tensor(v)
                else:
                    # skip empty lists
                    continue
            else:
                try:
                    out_ex[k] = torch.stack(v)
                except TypeError:
                    # can't stack string, etc. -- just leave em
                    extra_keys.append(k)

        return dict(out_ex)


class DocumentQueryCollatorWithPadding(transformers.DataCollatorWithPadding):
    tokenizer: transformers.PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    # TODO: Fix to use separate tokenizers

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        document_batch, query_batch, hn_document_batch, dataset_batch = [], [], [], []
        other_features = collections.defaultdict(list)
        for ex in features:
            doc_ex = {}
            hn_doc_ex = {}
            query_ex = {}
            dataset_ex = {}
            for k,v in ex.items():
                if is_doc(k):
                    doc_ex[k.replace('document_', '')] = v
                elif is_hn_doc(k):
                    hn_doc_ex[k.replace('negative_document_', '')] = v
                elif is_query(k):
                    query_ex[k.replace('query_', '')] = v
                elif is_dataset_doc(k):
                    dataset_ex[k.replace('dataset_', '')] = v
                else:
                    other_features[k].append(v)
            if len(doc_ex):
                document_batch.append(doc_ex)
            if len(hn_doc_ex):
                # handle multiple hard negatives
                n = len(next(iter(hn_doc_ex.values())))
                for i in range(n):
                    hn_document_batch.append({k: v[i] for k,v in hn_doc_ex.items() })
            if len(query_ex):
                query_batch.append(query_ex)
            if len(dataset_ex):
                dataset_batch.append(dataset_ex)
        
        # stack other features
        for k,v in other_features.items():
            if isinstance(v, list) and isinstance(v[0], int):
                other_features[k] = torch.tensor(v)
            else:
                try:
                    other_features[k] = torch.stack(v)
                except TypeError:
                    # TypeError: expected Tensor as element 0 in argument 0, but got list
                    # pad everything with -1s
                    pad_func = functools.partial(pad_tensor_to_length, length=self.max_length, value=-1)
                    other_features[k] = torch.stack(list(map(pad_func, v)))
        
        # pad stuff
        ex = {}
        ex.update(other_features)

        # pad_batch_func = functools.partial(
        #     self.tokenizer.pad,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.max_length,
        #     return_tensors=self.return_tensors,
        # )
        def pad_batch_func(ex_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            M = self.max_length
            # list of examples containing input ids & att mask
            output = collections.defaultdict(list)
            for ex in ex_list:
                output['input_ids'].append(ex['input_ids'][:M])
                output['attention_mask'].append(ex['attention_mask'][:M])
            output['input_ids'] = torch.nn.utils.rnn.pad_sequence(
                output['input_ids'], batch_first=True, padding_value=0)
            output['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
                output['attention_mask'], batch_first=True, padding_value=0
            )
            return output

        # tokenize documents.
        if len(document_batch):
            document_batch: Dict[str, torch.Tensor] = pad_batch_func(document_batch)
            # document_batch = cut_padding(document_batch, self.tokenizer.pad_token_id)
            ex.update({f'document_{k}': v for k,v in document_batch.items()})
        
        # tokenize hard negative documents.
        if len(hn_document_batch):
            hn_document_batch: Dict[str, torch.Tensor] = pad_batch_func(hn_document_batch)
            # hn_document_batch = cut_padding(hn_document_batch, self.tokenizer.pad_token_id)
            ex.update({f'negative_document_{k}': v for k,v in hn_document_batch.items()})

        # tokenize queries.
        if len(query_batch):
            query_batch: Dict[str, torch.Tensor] = pad_batch_func(query_batch)
            # query_batch = cut_padding(query_batch, self.tokenizer.pad_token_id)
            ex.update({f'query_{k}': v for k,v in query_batch.items()})

        # tokenize dataset-level documents.
        if len(dataset_batch):
            dataset_batch: Dict[str, torch.Tensor] = pad_batch_func(dataset_batch)
            # dataset_batch = cut_padding(dataset_batch, self.tokenizer.pad_token_id)
            ex.update({f'dataset_{k}': v for k,v in dataset_batch.items()})
        
        return ex
        
        
