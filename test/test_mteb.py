import os
import tempfile

from cde.lib.embed import TwoStageDenseEncoder
from cde.lib.eval.mteb import MTEB

from helpers import FakeDatasetTransformer


# os.environ["TOKENIZERS_PARALLELISM"] = "1"


gtr_mn = "sentence-transformers/gtr-t5-base"


def test_mteb_retrieval():
    """Test that MTEB uses the proper classes for retrieval,
    so that we can override things.
    """
    mteb = MTEB(
        tasks=["NFCorpus"], 
        task_langs=["en"],
        embedder_rerank="sentence-transformers/gtr-t5-base",
    )
    transductive_model = FakeDatasetTransformer()

    mteb_encoder = TwoStageDenseEncoder(
        model_name_or_path=gtr_mn,
        max_seq_length=16,
        encoder=transductive_model,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        results = mteb.run(
            mteb_encoder, 
            output_folder=os.path.join(temp_dir, "results_mteb", "test"),
            batch_size=1024, 
            corpus_chunk_size=100_000,
            verbosity=2
        )
    print(results)

    breakpoint()