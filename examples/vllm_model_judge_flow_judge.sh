

# = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# These environment variables are required for our server to run
# You can remove them on your local server
echo "Setting up environment variables"
echo "Warning: These environment variables are required for our server to run. You can remove them on your local server"
unset http_proxy https_proxy HF_ENDPOINT
# = = = = = = = = = = = = = = = = = = = = = = = = = = = =


export CUDA_VISIBLE_DEVICES=0
port=5001

# Workaround for vLLM 0.10.x bug: KeyError in _compute_kwargs docstring parser
# Monkey-patch to use .get() with empty default, then launch via CLI entrypoint
python -c "
import functools
import vllm.engine.arg_utils as au

_orig = au._compute_kwargs.__wrapped__

@functools.lru_cache(maxsize=30)
def _patched(cls):
    cls_docs = au.get_attr_docs(cls)
    # Wrap in defaultdict so missing keys return empty string
    import collections
    cls_docs_safe = collections.defaultdict(str, cls_docs)
    # Temporarily replace get_attr_docs to return the safe version
    orig_get = au.get_attr_docs
    au.get_attr_docs = lambda c: cls_docs_safe
    try:
        return _orig(cls)
    finally:
        au.get_attr_docs = orig_get

au._compute_kwargs = _patched

import sys
sys.argv = ['vllm', 'serve', 'flowaicom/Flow-Judge-v0.1-AWQ',
    '--quantization', 'awq_marlin',
    '--port', '$port',
    '--tensor-parallel-size', '1',
    '--max-model-len', '8192',
    '--gpu-memory-utilization', '0.1',
    '--disable-log-requests',
    '--disable-log-stats']

from vllm.entrypoints.cli.main import main
main()
" &

echo "Started Flow-Judge server on port $port"


# Test with:
# curl http://localhost:5001/v1/models
