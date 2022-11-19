
__all__ = [
    'tree_flatten',
    'tree_unflatten',
    'dict_iterated_getitem',
    'dict_iterated_setitem',
    'set_statistics',
    "dict_get_list_of_items",
    "re_parse_float",
    "parse_kv_from_string",
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
    """ Get dictionary item successively.
        
        `ks`
            ['k1','k2'] or 'k1.k2'
    """
    if isinstance(ks, str):
        ks = ks.split('.')
    x = d
    for k in ks:
        x = x[k]
    return x
            
    
def dict_iterated_setitem(d, ks, v):
    x = d
    for k in ks[:-1]:
        x = x[k]
    x[ks[-1]] = v


def set_statistics(s1, s2):
    """ Computes size of s1&s2, s1-s2, s2-s1. """
    s1 = set(s1)
    s2 = set(s2)
    info = [
        ('s1', len(s1)),
        ('s2', len(s2)),
        ('s1&s2', len(s1&s2)),
        ('s1-s2', len(s1-s2)),
        ('s2-s1', len(s2-s1)),
    ]
    for x, y in info:
        print(f'{x:6} {y}')


def dict_get_list_of_items(K, L, Q):
    """Given a dictionary `d` decomposed to two lists `{k:v for k, v in zip(K, L)}`,
        and a list of query keys `KQ`, return a list of corresponding values in `d`."""
    container_cls = type(L)
    indices = [K.index(k) for k in Q]
    return container_cls(L[i] for i in indices)


def re_parse_float(s, k, v=None):
    """Parse string `s` for floating 'x' of form 'k=x' with defaut value `v`."""
    import re
    m = re.search(k+'=([+-]?[0-9]+([.][0-9]*)?|[.][0-9]+)', s)
    x = float(m.groups()[0]) if m else v
    return x


def parse_kv_from_string(s):
    kvs = []
    for i, segment in enumerate(s.split('_')):
        if '=' in segment:
            k, v = segment.split('=')
            try:
                v = float(v)
            except:
                pass
            kvs.append((k, v))
        else:
            kvs.append((i, segment))
    d = dict(kvs)
    return d