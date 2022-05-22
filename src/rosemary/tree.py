
__all__ = [
    'tree_flatten',
    'tree_unflatten',
    'dict_iterated_getitem',
    'dict_iterated_setitem',
]


def tree_flatten(tree, parent_key='', sep='.'):
    """`tree` is a dictionary"""
    import collections
    try:
        collectionsAbc = collections.abc
    except AttributeError:
        collectionsAbc = collections

    items = []
    for k, v in tree.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collectionsAbc.Mapping):
            items.extend(tree_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def tree_unflatten(flat):
    """Nest depth-1 tree to nested tree 
            {'a.b.c': x} -> {'a': {'b': {'c': x}}} """
    def recursion(k, v, out):
        k, *rest = k.split('.', 1)
        if rest:
            recursion(rest[0], v, out.setdefault(k, {}))
        else:
            out[k] = v
    d = {}
    for k, v in flat.items():
        recursion(k, v, d)
    return d
    

def dict_iterated_getitem(d, ks):
    x = d
    for k in ks:
        x = x[k]
    return x
            
    
def dict_iterated_setitem(d, ks, v):
    x = d
    for k in ks[:-1]:
        x = x[k]
    x[ks[-1]] = v