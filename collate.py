from typing import Any, Dict, List, Optional, Union

import collections
import functools

import torch
import transformers


is_doc = lambda s: s.startswith('document_')
is_hn_doc = lambda s: s.startswith('negative_document_')
is_query = lambda s: s.startswith('query_')


def pad_tensor_to_length(the_list: List[int], length: int = 0, value: int = 0) -> List[int]:
    num_pads = length - len(the_list)
    if num_pads == 0:
        return torch.tensor(the_list, dtype=torch.long)
    else:
        return torch.tensor(the_list + [value] * num_pads, dtype=torch.long)


def cut_padding(batch: Dict[str, torch.Tensor], pad_token: int) -> Dict[str, torch.Tensor]:
    """Truncates doc if some stuff is all padding at the end."""
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    # 
    b, s = batch['input_ids'].shape
    # 
    all_padding = (batch['input_ids'] == pad_token).all(dim=0)
    if all_padding.sum() == 0:
        return batch
    # 
    padding_start = all_padding.int().argmax()
    batch['input_ids'] = batch['input_ids'][:, :padding_start]
    batch['attention_mask'] = batch['attention_mask'][:, :padding_start]
    return batch

class DocumentQueryCollatorWithPadding(transformers.DataCollatorWithPadding):
    tokenizer: transformers.PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        document_batch, query_batch, hn_document_batch = [], [], []
        other_features = collections.defaultdict(list)
        for ex in features:
            doc_ex = {}
            hn_doc_ex ={}
            query_ex = {}
            for k,v in ex.items():
                if is_doc(k):
                    doc_ex[k.replace('document_', '')] = v
                elif is_hn_doc(k):
                    hn_doc_ex[k.replace('negative_document_', '')] = v
                elif is_query(k):
                    query_ex[k.replace('query_', '')] = v
                else:
                    other_features[k].append(v)
            if len(doc_ex):
                document_batch.append(doc_ex)
            if len(hn_doc_ex):
                hn_document_batch.append(hn_doc_ex)
            if len(query_ex):
                query_batch.append(query_ex)
        
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
                    max_length = max(map(len, v))
                    pad_func = functools.partial(pad_tensor_to_length, length=max_length, value=-1)
                    other_features[k] = torch.stack(list(map(pad_func, v)))
        
        # pad stuff
        ex = {}
        ex.update(other_features)

        # tokenize documents.
        if len(document_batch):
            document_batch: Dict[str, torch.Tensor] = self.tokenizer.pad(
                document_batch,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            document_batch = cut_padding(document_batch, self.tokenizer.pad_token_id)
            ex.update({f'document_{k}': v for k,v in document_batch.items()})
        
        # tokenize hard negative documents.
        if len(hn_document_batch):
            hn_document_batch: Dict[str, torch.Tensor] = self.tokenizer.pad(
                hn_document_batch,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            hn_document_batch = cut_padding(hn_document_batch, self.tokenizer.pad_token_id)
            ex.update({f'negative_document_{k}': v for k,v in hn_document_batch.items()})

        # tokenize queries.
        if len(query_batch):
            query_batch: Dict[str, torch.Tensor] = self.tokenizer.pad(
                query_batch,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            query_batch = cut_padding(query_batch, self.tokenizer.pad_token_id)
            ex.update({f'query_{k}': v for k,v in query_batch.items()})
        
        return ex
        
        