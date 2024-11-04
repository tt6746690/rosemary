from joblib import Parallel, delayed

__all__ = [
    "joblib_parallel_process",
]

def joblib_parallel_process(fn, iterable, n_jobs, prefer=None, use_tqdm=False):
    """Computes [fn(x) for x in iterable] with `n_jobs` number of processes.
            Setting `use_tqdm` to True implicitly converts `iterable` to list.

        Sometimes setting `backend='loky'` makes too many cores running.
            `backend='multiprocessing'` seems to be less CPU compute intensive.
    """
    parallel_execute = Parallel(n_jobs=n_jobs,
                                prefer=prefer)
    delayed_fn = delayed(fn)
    if use_tqdm:
        from tqdm import tqdm
        desc = use_tqdm if isinstance(use_tqdm, str) else ""
        try:
            iterable = tqdm(iterable, total=len(iterable), desc=desc)
        except TypeError:
            iterable = tqdm(list(iterable), desc=desc)
    result = parallel_execute(delayed_fn(x) for x in iterable)
    return result