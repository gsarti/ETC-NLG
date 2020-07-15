def embeddings_from_file(file, model_name):
    if args.language == "it":
        we_model = CamemBERT(model_name)
    else:
        we_model = RoBERTa(model_name)
    pooling = Pooling(we_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[we_model, pooling])
    with open(file) as file:
        train_text = list(map(lambda x: x, file.readlines()))
        return np.array(model.encode(train_text))