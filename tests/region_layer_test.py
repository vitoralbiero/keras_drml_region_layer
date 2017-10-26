from keras import backend as K
from layers import RegionLayer
from keras import layers
import numpy as np
from unittest import TestCase
from unittest.mock import Mock, call


class TestRegionLayer(TestCase):

    def tearDown(self):
        K.clear_session()

    def test_split_region(self):
        # some RGB input image with size 100x100
        l1 = layers.Input(shape=(100, 100, 3))
        region_layer = RegionLayer()
        region_layer.split(l1, n_rows=5, n_cols=5)
        for region in region_layer._regions:
            self.assertTrue(region._keras_shape == (None, 20, 20, 3))

    def test_split_input_is_a_4D_tensor(self):
        l1 = layers.Input(shape=(100, 100))
        region_layer = RegionLayer()
        self.assertRaises(ValueError, region_layer.split, l1, 5, 5)

    def test_split_input_height_is_divisible(self):
        l1 = layers.Input(shape=(99, 100, 3))
        region_layer = RegionLayer()
        self.assertRaises(ValueError, region_layer.split, l1, 5, 5)

    def test_split_input_width_is_divisible(self):
        l1 = layers.Input(shape=(100, 99, 3))
        region_layer = RegionLayer()
        self.assertRaises(ValueError, region_layer.split, l1, 5, 5)

    def test_concatenate_regions(self):
        # some RGB input image with size 80x90
        l1 = layers.Input(shape=(80, 90, 3))
        region_layer = RegionLayer()
        region_layer.split(l1, n_rows=2, n_cols=3)
        l2 = region_layer.concatenate_convolution()

        data_input = np.float32(np.random.normal(size=(1, 80, 90, 3)))
        get_output = K.function([l1], [l2])
        output = get_output([data_input])[0]
        np.testing.assert_equal(data_input, output)

    def test_concatenate_regions_with_axis(self):
        # some RGB input image with size 32x32
        l1 = layers.Input(shape=(32, 32, 3))
        region_layer = RegionLayer()
        region_layer.split(l1, n_rows=2, n_cols=2)
        # this is a "mock" for a fully connected layer
        region_layer.add(lambda x: layers.Flatten()(x))
        l2 = region_layer.concatenate_fully_connected()

        data_input = np.float32(np.random.normal(size=(2, 32, 32, 3)))
        get_output = K.function([l1], [l2])
        output = get_output([data_input])[0]
        split_1 = data_input[:, 0:16, 0:16, :].reshape((2, 16 * 16 * 3))
        split_2 = data_input[:, 0:16, 16:32, :].reshape((2, 16 * 16 * 3))
        split_3 = data_input[:, 16:32, 0:16, :].reshape((2, 16 * 16 * 3))
        split_4 = data_input[:, 16:32, 16:32, :].reshape((2, 16 * 16 * 3))
        concat = np.concatenate([split_1, split_2, split_3,
                                 split_4], axis=1)

        np.testing.assert_equal(concat, output)

    def test_add_operation_on_regions(self):
        # some RGB input image with size 80x90
        l1 = layers.Input(shape=(80, 90, 3))
        region_layer = RegionLayer()
        region_layer.split(l1, n_rows=2, n_cols=3)

        # clone regions
        regions = region_layer._regions[:]

        region_operation = Mock()
        region_layer.add(region_operation)

        expected_call_count = len(region_layer._regions)

        self.assertEqual(expected_call_count, region_operation.call_count)
        for region in regions:
            self.assertTrue(call(region) in region_operation.mock_calls)

        self.assertNotEqual(regions, region_layer._regions)
