# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test HyperAttention and all of its subroutines."""

from absl.testing import absltest
import jax
import jax.numpy as jnp

from hyperattention import hyper_attention

jax.config.update("jax_threefry_partitionable", False)


class HyperAttentionTest(absltest.TestCase):

  def test_softmax_attention_and_normalizer_without_mask(self):
    """Test softmax_attention_and_normalizer without mask nor causal."""
    q = jnp.arange(24).astype(jnp.float32).reshape(1, 2, 3, 4)
    k = jnp.arange(-24, 0).astype(jnp.float32).reshape(1, 2, 3, 4) / 24
    v = jnp.arange(12).astype(jnp.float32).reshape(1, 2, 3, 2)
    output, lse = hyper_attention.get_softmax_attention_and_normalizers(q, k, v)
    self.assertTrue(
        jnp.allclose(
            lse,
            jnp.array([[
                [[-1.02806366], [-6.37155157], [-11.33201719]],
                [[-2.69716354], [-3.5387341], [-4.37422781]],
            ]]),
            atol=2e-2,
        )
    )
    self.assertTrue(
        jnp.allclose(
            output,
            jnp.array([[
                [
                    [2.64031334, 3.64031334],
                    [3.64400968, 4.64400968],
                    [3.91245296, 4.91245296],
                ],
                [
                    [9.97754064, 10.97754064],
                    [9.99412635, 10.99412635],
                    [9.99845503, 10.99845503],
                ],
            ]]),
            atol=2e-2,
        )
    )

  def test_softmax_attention_and_normalizer_with_causal(self):
    """Test softmax_attention_and_normalizer with causal but no mask."""
    q = jnp.arange(24).astype(jnp.float32).reshape(1, 2, 3, 4)
    k = jnp.arange(-24, 0).astype(jnp.float32).reshape(1, 2, 3, 4) / 24
    v = jnp.arange(12).astype(jnp.float32).reshape(1, 2, 3, 2)
    output, lse = hyper_attention.get_softmax_attention_and_normalizers(
        q, k, v, causal=True
    )
    self.assertTrue(
        jnp.allclose(
            lse,
            jnp.array([[
                [[-2.70833333], [-8.22668367], [-11.33201719]],
                [[-11.70833333], [-9.37207598], [-4.37422781]],
            ]]),
            atol=2e-2,
        )
    )
    self.assertTrue(
        jnp.allclose(
            output,
            jnp.array([[
                [
                    [0.0, 1.0],
                    [1.72431669, 2.72431669],
                    [3.91245296, 4.91245296],
                ],
                [[6.0, 7.0], [7.9941605, 8.9941605], [9.99845503, 10.99845503]],
            ]]),
            atol=2e-2,
        )
    )

  def test_softmax_attention_and_normalizer_with_mask(self):
    """Test softmax_attention_and_normalizer with mask but no causal."""
    q = jnp.arange(24).astype(jnp.float32).reshape(1, 2, 3, 4)
    k = jnp.arange(-24, 0).astype(jnp.float32).reshape(1, 2, 3, 4) / 24
    v = jnp.arange(12).astype(jnp.float32).reshape(1, 2, 3, 2)
    mask = (
        jnp.ones(shape=(1, 2, 3, 3), dtype=jnp.bool_).at[0, 1, 2, 2].set(False)
    )
    output, lse = hyper_attention.get_softmax_attention_and_normalizers(
        q, k, v, mask=mask
    )
    self.assertTrue(
        jnp.allclose(
            lse,
            jnp.array([[
                [[-1.0281138], [-6.373765], [-11.337366]],
                [[-2.7007961], [-3.5435517], [-11.536829]],
            ]]),
            atol=2e-2,
        ),
    )
    self.assertTrue(
        jnp.allclose(
            output,
            jnp.array([[
                [
                    [2.6445312, 3.6455078],
                    [3.6464844, 4.647583],
                    [3.909668, 4.9091797],
                ],
                [
                    [9.971924, 10.9713745],
                    [9.984549, 10.983596],
                    [8.004601, 9.005367],
                ],
            ]]),
            atol=2e-2,
        )
    )

  def test_softmax_attention_and_normalizer_with_causal_and_mask(self):
    """Test softmax_attention_and_normalizer with both causal and mask."""
    q = jnp.arange(24).astype(jnp.float32).reshape(1, 2, 3, 4)
    k = jnp.arange(-24, 0).astype(jnp.float32).reshape(1, 2, 3, 4) / 24
    v = jnp.arange(12).astype(jnp.float32).reshape(1, 2, 3, 2)
    mask = (
        jnp.ones(shape=(1, 2, 3, 3), dtype=jnp.bool_).at[0, 1, 2, 2].set(False)
    )
    output, lse = hyper_attention.get_softmax_attention_and_normalizers(
        q, k, v, causal=True, mask=mask
    )
    self.assertTrue(
        jnp.allclose(
            lse,
            jnp.array([[
                [[-2.7089844], [-8.223406], [-11.337366]],
                [[-11.708008], [-9.368667], [-11.536829]],
            ]]),
            atol=2e-2,
        ),
    )
    self.assertTrue(
        jnp.allclose(
            output,
            jnp.array([[
                [[0.0, 1.0], [1.7265625, 2.727539], [3.909668, 4.9091797]],
                [[6.0, 7.0], [7.9862366, 8.985245], [8.004601, 9.005367]],
            ]]),
            atol=2e-2,
        )
    )

  def test_add_self_attentions(self):
    """Test add_self_attentions to merge two attentions."""
    random_keys = jax.random.split(jax.random.PRNGKey(0), 7)
    query = jax.random.normal(random_keys[0], shape=(3, 4, 10, 8))
    key1 = jax.random.normal(random_keys[1], shape=(3, 4, 20, 8))
    value1 = jax.random.normal(random_keys[2], shape=(3, 4, 20, 4))
    mask1 = jax.random.bernoulli(random_keys[3], 0.5, shape=(3, 4, 10, 20))
    key2 = jax.random.normal(random_keys[4], shape=(3, 4, 15, 8))
    value2 = jax.random.normal(random_keys[5], shape=(3, 4, 15, 4))
    mask2 = jax.random.bernoulli(random_keys[6], 0.5, shape=(3, 4, 10, 15))
    output1, lse1 = hyper_attention.get_softmax_attention_and_normalizers(
        query, key1, value1, mask=mask1
    )
    output2, lse2 = hyper_attention.get_softmax_attention_and_normalizers(
        query, key2, value2, mask=mask2
    )
    output3, lse3 = hyper_attention.add_self_attentions(
        output1, lse1, output2, lse2
    )
    output4, lse4 = hyper_attention.get_softmax_attention_and_normalizers(
        query,
        jnp.concatenate([key1, key2], axis=-2),
        jnp.concatenate([value1, value2], axis=-2),
        mask=jnp.concatenate([mask1, mask2], axis=-1),
    )
    self.assertTrue(
        jnp.allclose(
            lse3,
            lse4,
            atol=2e-2,
        )
    )
    self.assertTrue(
        jnp.allclose(
            output3,
            output4,
            atol=2e-2,
        )
    )

  def test_simhash(self):
    """Test applying SimHash."""
    hash_func = hyper_attention.SimHash(
        dimension=2, num_projection=3, random_key=jax.random.PRNGKey(417)
    )
    # The sampled vectors are
    # (-0.7198806,  -1.0687262),
    # (0.95523006  0.96753764),
    # (-0.28749377, -0.65385664)
    x = jnp.array([
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ]).reshape(2, 2, 2, 2)
    self.assertTrue(
        jnp.allclose(
            hash_func.apply(x),
            jnp.array([[[5.0, 5.0], [2.0, 5.0]], [[2.0, 5.0], [2.0, 2.0]]]),
            atol=2e-2,
        )
    )

  def test_select_features_by_indices(self):
    """Test select_features_by_indices for data with ndim=4."""
    x = jnp.arange(24).reshape(2, 2, 3, 2)
    indices = jnp.array([[[1, 0], [2, 1]], [[2, 2], [0, 2]]])
    self.assertListEqual(
        hyper_attention.select_features_by_indices(x, indices).tolist(),
        jnp.array([
            [[[2, 3], [0, 1]], [[10, 11], [8, 9]]],
            [[[16, 17], [16, 17]], [[18, 19], [22, 23]]],
        ]).tolist(),
    )

  def test_hyper_attention_without_causal(self):
    """Test hyper_attention without causal."""
    q = jnp.arange(24).astype(jnp.float32).reshape(1, 2, 3, 4)
    k = jnp.arange(-24, 0).astype(jnp.float32).reshape(1, 2, 3, 4) / 24
    v = jnp.arange(12).astype(jnp.float32).reshape(1, 2, 3, 2)
    attention_mechanism = hyper_attention.HyperAttention(
        dimension=4,
        min_bucket_size=1,
        max_bucket_size=4,
        min_sample_size=1,
        max_sample_size=4,
        min_seq_len=1,
    )
    output, lse = attention_mechanism.attention_without_causal_mask(q, k, v)
    self.assertTrue(
        jnp.allclose(
            lse,
            jnp.array([[
                [[-0.9646351], [-7.9831605], [-17.708334]],
                [[-2.675549], [-9.375], [-10.443199]],
            ]]),
            atol=2e-2,
        ),
        lse,
    )
    self.assertTrue(
        jnp.allclose(
            output,
            jnp.array([[
                [
                    [1.9014053, 2.9014053],
                    [1.3516539, 2.3516538],
                    [0.0, 1.0],
                ],
                [[9.935492, 10.935492], [8.0, 9.0], [7.999486, 8.999486]],
            ]]),
            atol=2e-2,
        )
    )
    output2, lse2 = attention_mechanism.get_attention_and_normalizers(
        q, k, v, causal=False
    )
    self.assertTrue(
        jnp.allclose(
            lse,
            lse2,
            atol=2e-2,
        )
    )
    self.assertTrue(
        jnp.allclose(
            output,
            output2,
            atol=2e-2,
        )
    )

  def test_hyper_attention_with_causal(self):
    """Test hyper_attention with causal."""
    q = jnp.arange(64).astype(jnp.float32).reshape(1, 2, 8, 4)
    k = jnp.arange(-64, 0).astype(jnp.float32).reshape(1, 2, 8, 4) / 24
    v = jnp.arange(32).astype(jnp.float32).reshape(1, 2, 8, 2)
    attention_mechanism = hyper_attention.HyperAttention(
        dimension=4,
        min_bucket_size=1,
        max_bucket_size=4,
        min_sample_size=1,
        max_sample_size=4,
        min_seq_len=1,
    )
    output, lse = attention_mechanism.get_attention_and_normalizers(
        q, k, v, causal=True
    )
    self.assertTrue(
        jnp.allclose(
            lse,
            jnp.array([[
                [
                    [-7.7083335],
                    [-26.560041],
                    [-42.99699],
                    [-56.69729],
                    [-67.708336],
                    [-76.0409],
                    [-81.70793],
                    [-84.70828],
                ],
                [
                    [-85.04167],
                    [-82.708336],
                    [-77.708336],
                    [-70.04167],
                    [-59.708336],
                    [-46.708336],
                    [-31.041668],
                    [-12.708334],
                ],
            ]]),
            atol=2e-2,
        )
    )
    self.assertTrue(
        jnp.allclose(
            output,
            jnp.array([[
                [
                    [0.0, 1.0],
                    [1.7243173, 2.724317],
                    [3.905809, 4.905809],
                    [5.9780188, 6.9780183],
                    [7.999999, 8.999999],
                    [9.998457, 10.998458],
                    [11.999187, 12.999187],
                    [13.999892, 14.999892],
                ],
                [
                    [16.0, 17.0],
                    [17.999992, 18.999992],
                    [19.999998, 20.999996],
                    [22.0, 23.0],
                    [24.0, 25.0],
                    [26.0, 27.0],
                    [28.0, 29.0],
                    [30.0, 31.0],
                ],
            ]]),
            atol=2e-2,
        )
    )

  def test_hyper_attention_with_different_sample_ratios(self):
    """Test hyper_attention with different sample ratios."""
    q = jnp.arange(64).astype(jnp.float32).reshape(1, 2, 8, 4)
    k = jnp.arange(-64, 0).astype(jnp.float32).reshape(1, 2, 8, 4) / 24
    v = jnp.arange(32).astype(jnp.float32).reshape(1, 2, 8, 2)
    attention_mechanism = hyper_attention.HyperAttention(
        dimension=4,
        min_bucket_size=1,
        max_bucket_size=4,
        bucket_size_ratio=1.0 / 4.0,
        min_sample_size=1,
        max_sample_size=4,
        sample_size_ratio=1.0 / 4.0,
        min_seq_len=1,
    )
    output, lse = attention_mechanism.get_attention_and_normalizers(q, k, v)
    self.assertTrue(
        jnp.allclose(
            lse,
            jnp.array([[
                [
                    [-2.6933014],
                    [-15.397639],
                    [-33.500385],
                    [-49.770912],
                    [-69.2631],
                    [-74.9175],
                    [-91.30825],
                    [-104.37495],
                ],
                [
                    [-6.8749857],
                    [-7.7083297],
                    [-75.59929],
                    [-84.43261],
                    [-60.265945],
                    [-66.43262],
                    [-34.275845],
                    [-37.768566],
                ],
            ]]),
            atol=2e-2,
        ),
    )
    self.assertTrue(
        jnp.allclose(
            output,
            jnp.array([[
                [
                    [11.145837, 12.145836],
                    [13.166116, 14.166116],
                    [9.919122, 10.919122],
                    [10.0, 11.0],
                    [10.0, 11.0],
                    [9.999999, 11.0],
                    [10.0, 11.0],
                    [9.999892, 10.999892],
                ],
                [
                    [29.999971, 30.99997],
                    [29.999994, 30.999994],
                    [28.0, 29.0],
                    [28.0, 29.0],
                    [28.0, 29.0],
                    [28.0, 29.0],
                    [28.0, 29.0],
                    [28.0, 29.0],
                ],
            ]]),
            atol=2e-2,
        ),
    )

  def test_hyper_attention_with_local_attention_only(self):
    """Test hyper_attention without SortingLSH nor sampling."""
    q = jnp.arange(64).astype(jnp.float32).reshape(1, 2, 8, 4)
    k = jnp.arange(-64, 0).astype(jnp.float32).reshape(1, 2, 8, 4) / 24
    v = jnp.arange(32).astype(jnp.float32).reshape(1, 2, 8, 2)
    attention_mechanism = hyper_attention.HyperAttention(
        dimension=4,
        min_bucket_size=1,
        max_bucket_size=4,
        bucket_size_ratio=1.0,
        min_sample_size=1,
        max_sample_size=4,
        sample_size_ratio=1.0,
        min_seq_len=1,
        use_sorting=False,
        use_sampling=False,
    )
    output, lse = attention_mechanism.get_attention_and_normalizers(
        q, k, v, causal=True
    )
    self.assertTrue(
        jnp.allclose(
            lse,
            jnp.array([[
                [
                    [-7.7083335],
                    [-26.560041],
                    [-42.998833],
                    [-56.697166],
                    [-67.70553],
                    [-76.0409],
                    [-81.70814],
                    [-84.70828],
                ],
                [
                    [-85.04167],
                    [-82.708336],
                    [-77.708336],
                    [-70.04167],
                    [-59.708336],
                    [-46.708336],
                    [-31.041668],
                    [-12.708334],
                ],
            ]]),
            atol=2e-2,
        )
    )
    self.assertTrue(
        jnp.allclose(
            output,
            jnp.array([[
                [
                    [0.0, 1.0],
                    [1.7243173, 2.724317],
                    [3.9128563, 4.9128766],
                    [5.9775343, 6.977534],
                    [7.994365, 8.994363],
                    [9.998455, 10.998455],
                    [11.999607, 12.999607],
                    [13.999892, 14.999892],
                ],
                [
                    [16.0, 17.0],
                    [17.999992, 18.999992],
                    [19.999998, 20.999998],
                    [22.0, 23.0],
                    [23.999998, 24.999998],
                    [26.0, 27.0],
                    [28.0, 29.0],
                    [30.0, 31.0],
                ],
            ]]),
            atol=2e-2,
        )
    )

  def test_hyper_attention_with_large_min_seq_len(self):
    """Test hyper_attention with large min_seq_len."""
    q = jnp.arange(64).astype(jnp.float32).reshape(1, 2, 8, 4)
    k = jnp.arange(-64, 0).astype(jnp.float32).reshape(1, 2, 8, 4) / 24
    v = jnp.arange(32).astype(jnp.float32).reshape(1, 2, 8, 2)
    attention_mechanism = hyper_attention.HyperAttention(
        dimension=4,
        min_bucket_size=1,
        max_bucket_size=4,
        min_sample_size=1,
        max_sample_size=4,
        min_seq_len=8,
    )
    output, lse = attention_mechanism.get_attention_and_normalizers(
        q, k, v, causal=True
    )
    self.assertTrue(
        jnp.allclose(
            lse,
            jnp.array([[
                [
                    [-7.7109375],
                    [-26.564053],
                    [-42.973866],
                    [-56.730213],
                    [-67.70423],
                    [-76.06952],
                    [-81.67559],
                    [-84.70698],
                ],
                [
                    [-85.08594],
                    [-82.66016],
                    [-77.708984],
                    [-70.01172],
                    [-59.740234],
                    [-46.708008],
                    [-31.031738],
                    [-12.725708],
                ],
            ]]),
            atol=1e-1,
        ),
    )
    self.assertTrue(
        jnp.allclose(
            output,
            jnp.array([[
                [
                    [0.0, 1.0],
                    [1.7265625, 2.7265625],
                    [3.9223633, 4.924225],
                    [5.9770584, 6.9772468],
                    [7.98554, 8.984435],
                    [10.006351, 11.007145],
                    [12.001907, 13.002098],
                    [14.000667, 15.000722],
                ],
                [
                    [16.0, 17.0],
                    [18.000053, 19.000057],
                    [20.00002, 21.00002],
                    [22.000006, 23.000006],
                    [24.000002, 25.000002],
                    [26.0, 27.0],
                    [28.0, 29.0],
                    [30.0, 31.0],
                ],
            ]]),
            atol=1e-1,
        )
    )


if __name__ == "__main__":
  absltest.main()
