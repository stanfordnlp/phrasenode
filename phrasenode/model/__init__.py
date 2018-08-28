def create_model(config):
    model_name = config.model.name
    if model_name == 'encoding':
        # Dot product between node embedding and phrase embedding
        node_embedder = create_node_embedder(config)
        from phrasenode.model.encoding import get_encoding_model
        model = get_encoding_model(config, node_embedder)
    elif model_name == 'alignment':
        # Dot product between node embedding and phrase embedding
        node_embedder = create_node_embedder(config)
        from phrasenode.model.alignment import get_alignment_model
        model = get_alignment_model(config, node_embedder)
    elif model_name == 'ensemble':
        # Encoding + Alignment
        node_embedder = create_node_embedder(config)
        from phrasenode.model.ensemble import get_ensemble_model
        model = get_ensemble_model(config, node_embedder)
    else:
        raise ValueError('Unknown model name {}'.format(model_name))
    return model


def create_node_embedder(config):
    node_embedder_name = config.model.node_embedder.name
    if node_embedder_name == 'stupid':
        from phrasenode.node_embedder.stupid import get_stupid_embedder
        node_embedder = get_stupid_embedder(config)
    elif node_embedder_name == 'proppy':
        from phrasenode.node_embedder.proppy import get_proppy_embedder
        node_embedder = get_proppy_embedder(config)
    elif node_embedder_name == 'allan':
        from phrasenode.node_embedder.allan import get_allan_embedder
        node_embedder = get_allan_embedder(config)
    else:
        raise ValueError('Unknown node embedder {}'.format(node_embedder_name))
    return node_embedder
