import pytest
import torch
import transformers

from tqdm.autonotebook import tqdm


def _test_context_variance_st(model):
    wikipedia_contexts = {
        "pet": "A pet, or companion animal, is an animal kept primarily for a person's company or entertainment rather than as a working animal, livestock, or a laboratory animal.",
        "nuclearphysics": "Nuclear physics is the field of physics that studies atomic nuclei and their constituents and interactions, in addition to the study of other forms of nuclear matter."
    }

    all_doc_embeddings = []
    textset = ["I tried to use your model by using various context texts to get dataset_embeddings, and then embed my strings to obtain doc_embeddings under each dataset_embeddings. As follows:"]
    minicorpus_size = 512
    dataset_embeddings_list = model.encode(
        list(wikipedia_contexts.values()),
        prompt_name="document",
        convert_to_tensor=True,
    )
    for i in range(2):
        # 3. First stage: embed the context docs
        dataset_embeddings = dataset_embeddings_list[None, i].expand(minicorpus_size, -1)
        
        # 4. Second stage: embed the docs
        doc_embeddings = model.encode(
            textset,
            prompt_name="document",
            dataset_embeddings=dataset_embeddings,
            convert_to_tensor=True,
        )
        all_doc_embeddings.append(doc_embeddings)
    
    assert len(all_doc_embeddings) == 2
    first_emb, second_emb = all_doc_embeddings
    distance_between_them = (first_emb - second_emb).norm(p=2)
    
    breakpoint()
    assert distance_between_them > 0.1
    print(f"Got distance based on context:", distance_between_them)


@torch.no_grad
def _test_context_variance_hf(model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
    # query_prefix = "search_query: "
    if not torch.cuda.is_available():
        # skip with pytest
        pytest.skip("CUDA is not available")

    model.cuda()

    document_prefix = "search_document: "

    # first stage
    minicorpus_size = model.config.transductive_corpus_size
    minicorpus_docs_1 = ["test string 1"] * minicorpus_size
    minicorpus_docs_2 = ["test string 2"] * minicorpus_size

    assert len(minicorpus_docs_1) == len(minicorpus_docs_2) == minicorpus_size
    minicorpus_docs_1 = tokenizer(
        [document_prefix + doc for doc in minicorpus_docs_1],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    ).to(model.device)
    minicorpus_docs_2 = tokenizer(
        [document_prefix + doc for doc in minicorpus_docs_2],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    ).to(model.device)

    batch_size = 32
    dataset_embeddings_1 = []
    dataset_embeddings_2 = []
    for i in tqdm(range(0, len(minicorpus_docs_1["input_ids"]), batch_size)):
        minicorpus_docs_batch_1 = { k: v[i:i+batch_size] for k,v in minicorpus_docs_1.items() }
        minicorpus_docs_batch_2 = { k: v[i:i+batch_size] for k,v in minicorpus_docs_2.items() }
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            dataset_embeddings_1.append(
                model.first_stage_model(**minicorpus_docs_batch_1)
            )
            dataset_embeddings_2.append(
                model.first_stage_model(**minicorpus_docs_batch_2)
            )

    dataset_embeddings_1 = torch.cat(dataset_embeddings_1)
    dataset_embeddings_2 = torch.cat(dataset_embeddings_2)

    # second stage
    docs = ["first test document", "second test document", "third test document", "fourth test document", "fifth test document"]
    docs = tokenizer(
        [document_prefix + doc for doc in docs],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        doc_embeddings_1 = model.second_stage_model(
            input_ids=docs["input_ids"],
            attention_mask=docs["attention_mask"],
            dataset_embeddings=dataset_embeddings_1,
        )
        doc_embeddings_2 = model.second_stage_model(
            input_ids=docs["input_ids"],
            attention_mask=docs["attention_mask"],
            dataset_embeddings=dataset_embeddings_2,
        )
    doc_embeddings_1 /= doc_embeddings_1.norm(p=2, dim=1, keepdim=True)
    doc_embeddings_2 /= doc_embeddings_2.norm(p=2, dim=1, keepdim=True)
    
    breakpoint()
    assert not torch.allclose(doc_embeddings_1, doc_embeddings_2, atol=1e-3)

def test_model():
    model = transformers.AutoModel.from_pretrained("jxm/cde-small-v1", trust_remote_code=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    _test_context_variance_hf(model=model, tokenizer=tokenizer)

def test_model_st():
    import sentence_transformers
    model = sentence_transformers.SentenceTransformer("jxm/cde-small-v1", trust_remote_code=True)
    _test_context_variance_st(model=model)


def test_model_local_hf():
    from cde.lib.hf_model import ContextualDocumentEmbeddingTransformer
    model = ContextualDocumentEmbeddingTransformer.from_pretrained("jxm/cde-small-v1")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    _test_context_variance_hf(model=model, tokenizer=tokenizer)
