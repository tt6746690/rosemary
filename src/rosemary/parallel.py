from joblib import Parallel, delayed

__all__ = [
    "joblib_parallel_process",
]

def joblib_parallel_process(fn, iterable, n_jobs, use_tqdm=False):
    """Computes [fn(x) for x in iterable] with `n_jobs` number of processes.
            Setting `use_tqdm` to True implicitly converts `iterable` to list.
    """
    parallel_execute = Parallel(n_jobs=n_jobs,
                                backend='multiprocessing',
                                prefer='processes')
    delayed_fn = delayed(fn)
    if use_tqdm:
        from tqdm import tqdm
        iterable = tqdm(list(iterable))
    tasks = iter(delayed_fn(x) for x in iterable)
    result = parallel_execute(tasks)
    return result