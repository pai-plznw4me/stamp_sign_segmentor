from unittest import TestCase
from dataprovider import StampSignSegmentDataprovider
import numpy as np


class TestStampSignSegmentDataprovider(TestCase):
    def setUp(self):
        self.sample_image = np.zeros([100, 100, 3])
        self.gt1_coord = [0, 0, 25, 25]
        self.gt2_coord = [75, 75, 100, 100]
        self.gt = np.array([self.gt1_coord, self.gt2_coord])

    def test_generate_segmentation_dataset(self):
        batch_xs, batch_ys = StampSignSegmentDataprovider.generate_segmentation_dataset(self.sample_image, self.gt)

        doc_h, doc_w = self.sample_image.shape[:2]
        # offset 이 적용된 좌표는 이미지 크기 내에 있어야 합니다.
        assert np.all(batch_ys[:, [0, 2]] >= 0) & np.all(batch_ys[:, [0, 2]] <= doc_w)
        assert np.all(batch_ys[:, [1, 3]] >= 0) & np.all(batch_ys[:, [1, 3]] <= doc_h)
