"""Keep actual training duration distinct from the optional LR horizon."""


def decision_schedule_updates(training, *, epochs, blocks, updates_per_collection):
    horizon = int(training.get('lr_schedule_epochs', epochs))
    warmup_epochs = int(training['warmup_epochs'])
    if min(epochs, blocks, updates_per_collection) < 1:
        raise ValueError('training duration and update counts must be positive')
    if horizon < epochs or not 0 <= warmup_epochs <= horizon:
        raise ValueError('LR horizon must cover training and warmup')
    per_epoch = blocks * updates_per_collection
    return horizon, per_epoch * horizon, per_epoch * warmup_epochs


def matched_cosine_endpoint_ratio(training, *, epochs, reference_horizon,
                                  blocks, updates_per_collection):
    """Floor for a shorter cosine that ends at the longer schedule's stop LR."""
    from .sft import cosine_learning_rate
    _, total, warmup = decision_schedule_updates(
        dict(training, lr_schedule_epochs=reference_horizon), epochs=epochs,
        blocks=blocks, updates_per_collection=updates_per_collection,
    )
    if epochs <= int(training['warmup_epochs']):
        raise ValueError('endpoint matching requires a post-warmup stop')
    return cosine_learning_rate(
        1., update=epochs * blocks * updates_per_collection, total_updates=total,
        warmup_updates=warmup, eta_min_ratio=float(training['cosine_eta_min_ratio']),
    )
