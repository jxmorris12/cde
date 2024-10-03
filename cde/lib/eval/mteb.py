TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "HotpotQA",
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FiQA2018",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
    "FEVER",
    "MSMARCO",
]

# TASK_LIST_RETRIEVAL = ["SCIDOCS", "SciFact", "NFCorpus", "TRECCOVID", "Touche2020"] # Small datasets.
# TASK_LIST_RETRIEVAL = ["TRECCOVID"]
# TASK_LIST_RETRIEVAL = ["FiQA2018"]


# TASK_LIST_RETRIEVAL = [
#     "ArguAna",
#     "NFCorpus", 
#     "SCIDOCS", 
#     "TRECCOVID", 
#     "SciFact", 
#     "FiQA2018", 
#     "Touche2020",
# ] # Small datasets.


# TASK_LIST_RETRIEVAL = ["QuoraRetrieval"]
# TASK_LIST_RETRIEVAL = ["NFCorpus"]
# TASK_LIST_RETRIEVAL = ["ArguAna"]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]

task2prefix_short = {}
for task in TASK_LIST_CLASSIFICATION:
    task2prefix_short[task] = {"query": "classification", "document": "classification"}

for task in TASK_LIST_CLUSTERING:
    task2prefix_short[task] = {"query": "clustering", "document": "clustering"}

for task in TASK_LIST_PAIR_CLASSIFICATION:
    task2prefix_short[task] = {"query": "classification", "document": "classification"}

for task in TASK_LIST_RERANKING:
    task2prefix_short[task] = {"query": "classification", "document": "classification"}

for task in TASK_LIST_RETRIEVAL:
    task2prefix_short[task] = {"query": "search_query", "document": "search_document"}

for task in TASK_LIST_STS:
    task2prefix_short[task] = {"query": "classification", "document": "classification"}

task2prefix_short["QuoraRetrieval"] = {"query": "search_query", "document": "search_query"}




TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)

task2prefix_long = {
    ########################## From BGE Training data ##############################################
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles.",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles.",
    "BiorxivClusteringP2P": "Identify the main and secondary category of Biorxiv papers based on the titles and abstracts.",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles.",
    "MedrxivClusteringP2P": "Identify the main and secondary category of Medrxiv papers based on the titles and abstracts.",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles.",
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts.",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic.",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles.",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs.",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles.",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts.",
    #########################################################################################################
    # From https://www.arxiv.org/abs/2409.15700
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual.",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise.",
    "Banking77Classification": "Given a online banking query, find the corresponding intents.",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category.",
    "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation.",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral.",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset.",
    #########################################################################################################
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query.",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question.",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question.",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question.",
    "ArguAna": "Given a claim, find documents that refute the claim.",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim.",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum.",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum.",
    "SciDocsRR": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper.",
    #########################################################################################################
    "CQADupstackAndroidRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackEnglishRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackGamingRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackGisRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackMathematicaRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackPhysicsRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackProgrammersRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackStatsRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackTexRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackUnixRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackWebmastersRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    "CQADupstackWordpressRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question.",
    #########################################################################################################
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim.",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia.",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question.",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question.",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper.",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim.",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question.",
    "TRECCOVID": "Given a query, retrieve documents that answer the query.",
    #########################################################################################################
    "BIOSSES": "Retrieve semantically similar text.",
    "SICK-R": "Retrieve semantically similar text.",
    "STS12": "Retrieve semantically similar text.",
    "STS13": "Retrieve semantically similar text.",
    "STS14": "Retrieve semantically similar text.",
    "STS15": "Retrieve semantically similar text.",
    "STS16": "Retrieve semantically similar text.",
    "STS17": "Retrieve semantically similar text.",
    "STS22": "Retrieve semantically similar text.",
    "STSBenchmark": "Retrieve semantically similar text.",
    "SummEval": "Given a news summary, retrieve other semantically similar summaries.",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet.",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet.",
    #########################################################################################################
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment.",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation.",
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum.",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history.",
}
assert len(set(task2prefix_long.keys()) & set(TASK_LIST)) == len(TASK_LIST)