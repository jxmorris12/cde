from spider.lib.eval.mteb import MTEB


def test_mteb_retrieval():
    """Test that MTEB uses the proper classes for retrieval,
    so that we can override things.
    """
    mteb = MTEB(tasks=["NFCorpus"], task_langs=["en"])

    breakpoint()