"""Cache-safe optimisation: KV snapshot / restore.

Each optimisation step runs an extra forward pass through the transformer at the
current scale. With KV caching active that pass would append to - and thereby
corrupt - the autoregressive cache that the real sampling pass depends on. The
snapshot is therefore taken once before the loop, restored before every
optimisation forward, and restored again in a `finally` block so that the real
sampling forward sees a clean cache exactly once.

When caching is off (`cached_k` is None everywhere) save/restore are no-ops and
the optimisation forward simply runs as a normal cache-free forward.
"""


def caching_is_active(infinity) -> bool:
    for m in infinity.modules():
        if m.__class__.__name__ == 'SelfAttention':
            return hasattr(m, 'cached_k') and m.cached_k is not None
    return False


def save_kv_cache(infinity) -> dict:
    cache = {}
    for name, m in infinity.named_modules():
        if m.__class__.__name__ == 'SelfAttention':
            cache[name] = {
                'k': m.cached_k.clone() if (hasattr(m, 'cached_k') and m.cached_k is not None) else None,
                'v': m.cached_v.clone() if (hasattr(m, 'cached_v') and m.cached_v is not None) else None,
            }
    return cache


def restore_kv_cache(infinity, cache: dict):
    for name, m in infinity.named_modules():
        if m.__class__.__name__ == 'SelfAttention' and name in cache:
            m.cached_k = cache[name]['k']
            m.cached_v = cache[name]['v']
