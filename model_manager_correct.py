import gc
import time

import h5py
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from alphafold3.model import features, model, params
from alphafold3.model.components import utils


class OptimizedModelRunner:
    """Compiled model runner for AF3Score inference.

    Two compiled functions are maintained:
      - no-init: pure diffusion from random noise (standard prediction)
      - with-init: diffusion seeded from input structure coordinates (scoring mode)

    The H5 files produced by 2_pdb2jax.py store coordinates as
    (num_samples, num_tokens, 24, 3) — already tiled, so they are used
    directly without further reshaping.
    """

    def __init__(self, config, device, model_dir):
        self.config = config
        self.device = device
        self.model_dir = model_dir

        self._model_params = params.get_model_haiku_params(model_dir=model_dir)
        self._prepare_compiled_models()

        self.inference_count = 0
        self.total_inference_time = 0
        self.first_inference_done = False

        self._position_cache = {}

    def _prepare_compiled_models(self):
        config = self.config

        @hk.transform
        def forward_fn_no_init(batch):
            return model.Model(config)(batch)

        @hk.transform
        def forward_fn_with_init(batch, init_positions):
            return model.Model(config)(batch, init_positions=init_positions)

        self._compiled_no_init = jax.jit(
            forward_fn_no_init.apply, device=self.device
        )
        self._compiled_with_init = jax.jit(
            forward_fn_with_init.apply, device=self.device
        )
        self._forward_fn_no_init = forward_fn_no_init
        self._forward_fn_with_init = forward_fn_with_init

    def _load_and_cache_positions(self, path: str) -> jnp.ndarray:
        """Load H5 coordinates (num_samples, num_tokens, 24, 3) onto device."""
        if path not in self._position_cache:
            h5_path = path if path.endswith('.h5') else path + '.h5'
            with h5py.File(h5_path, 'r') as f:
                array = np.array(f['coordinates'])
            self._position_cache[path] = jax.device_put(
                jnp.asarray(array, dtype=jnp.float32), self.device
            )
        return self._position_cache[path]

    @property
    def model_params(self):
        return self._model_params

    def run_inference(
        self,
        featurised_example: features.BatchDict,
        rng_key: jnp.ndarray,
        init_guess: bool = True,
        path: str = '',
    ) -> model.ModelResult:
        start_time = time.time()

        featurised_example = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
            ),
            self.device,
        )

        if not self.first_inference_done:
            print('First inference — triggering JAX compilation...')

        if init_guess and path:
            init_positions = self._load_and_cache_positions(path)
            result = self._compiled_with_init(
                self._model_params, rng_key, featurised_example, init_positions
            )
        else:
            result = self._compiled_no_init(
                self._model_params, rng_key, featurised_example
            )

        result = dict(result)
        result = jax.tree.map(np.asarray, result)
        result = jax.tree.map(
            lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
            result,
        )
        identifier = self._model_params['__meta__']['__identifier__'].tobytes()
        result['__identifier__'] = identifier

        elapsed = time.time() - start_time
        self.inference_count += 1
        self.total_inference_time += elapsed

        if not self.first_inference_done:
            print(f'First inference (incl. compilation): {elapsed:.2f}s')
            self.first_inference_done = True
        else:
            print(f'Inference #{self.inference_count}: {elapsed:.2f}s')

        return result

    def extract_structures(
        self,
        batch: features.BatchDict,
        result: model.ModelResult,
        target_name: str,
    ) -> list[model.InferenceResult]:
        return list(
            model.Model.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )

    def clear_cache(self):
        print('Clearing position cache...')
        self._position_cache.clear()
        gc.collect()


_global_runner = None


def get_optimized_runner(config, device, model_dir):
    global _global_runner
    if _global_runner is None:
        print('Initializing global model manager...')
        start_time = time.time()
        _global_runner = OptimizedModelRunner(config, device, model_dir)
        print(f'Global model init: {time.time() - start_time:.2f}s')
    return _global_runner
