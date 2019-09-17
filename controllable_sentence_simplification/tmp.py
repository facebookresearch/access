from ts.evaluation.general import get_easse_report_on_turkcorpus


if __name__ == '__main__':
    get_easse_report_on_turkcorpus()
    fairseq_train_and_evaluate()
    load_recommended_preprocessed_simplifier()
    prepare_wikilarge()
    prepare_turkcorpus_lower()
    prepare_fasttext_embeddings()
    prepare_turkcorpus()
    LevenshteinPreprocessor
    LengthRatioPreprocessor
    WordRankRatioPreprocessor
    DependencyTreeDepthRatioPreprocessor
    SentencePiecePreprocessor
