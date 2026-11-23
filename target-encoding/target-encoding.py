def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    cats = {}
    for cat, target in zip(categories, targets):
        if cat in cats:
            cats[cat].append(target)
        else:
            cats[cat] = [target]

    res = []
    for cat in categories:
        cat_t = cats[cat]
        res.append(sum(cat_t) / len(cat_t))

    return res