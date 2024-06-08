import warnings
    
def get_docs_with_query(db, query: str, num_of_docs: int, score_threshold: float = 0):
    '''
    Input
    -----
    Query: A string that will be used to calculate an embedding for search
    num_of_docs: An integer representing how many journal entries we want to retrieve
    score_threshold: A floating point value between 0 to 1 to filter the resulting set of retrieved docs. Deafult: 0s
    
    Output
    -----
    docs: a list of langchain Document Objects
    thresholds: a list of thresholds matching the document objects
    '''
    docs = db.similarity_search_with_relevance_scores(query, K=num_of_docs, score_threshold = score_threshold)
    
    if len(docs) == 0:
        warnings.warn("Warning: No documents were retrieved. consider lowering the score threshold")
        return [], []
    else:    
        docs_list, thresholds = zip(*docs)
        return docs_list, thresholds