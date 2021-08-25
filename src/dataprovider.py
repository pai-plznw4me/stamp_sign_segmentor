import os
from copy import copy
from glob import glob
from tensorflow.keras.utils import Sequence
from utils import random_offset, crop_images, search_img_paths, paths2imgs, copy_obj, plot_images
from utils import show_image, draw_rectangles
import numpy as np


class StampSignSegmentDataprovider(Sequence):
    def __init__(self, stamp_folder, docs_folder, batch_size, max_n_stamp=5):
        """
        Description:
         stamp, documentation 경로를 불러와 백터화합니다.(ndarray),
         모든 이미지는 RGB vector space 로 변형해 불러옵니다.

        :param stamp_folder: str, 도장 데이터들이 들어 있는 폴더
        :param docs_folder: str, 도장이 찍힐 문서 이미지가 들어 있는 폴더
        :param batch_size: int, __getitem__
        :param input_shape: tuple or list, 3차원의 모델에 입력될 이미지의 크기
         example)
          (370, 500, 3)
        :param onehot: bool, ys 값을 출력 할 때 onehot vector 로 출력할건지 아니면 cls 형태로 출력할건지 결정합니다.
        :param n_stamp_range: tuple, 문서 내 찍힐 도장의 개 수 범위
         example)
          (0, 10) ; 문서 내 도장이 0개 부터 10개 까지 찍힙니다.

        """

        self.stamp_folder = stamp_folder
        self.docs_folder = docs_folder
        self.batch_size = batch_size
        self.n_stamp_range = (1, max_n_stamp)

        # 지정된 경로에 있는 파일중 stamp 이미지 파일만 찾습니다.
        stamp_img_paths = glob(os.path.join(self.stamp_folder, '*'))
        stamp_img_paths = search_img_paths(stamp_img_paths)
        stamp_img_paths = sorted(stamp_img_paths)

        # 지정된 경로에 있는 파일중 documentatino 이미지 파일만 찾습니다.
        docus_img_paths = glob(os.path.join(self.docs_folder, '*'))
        docus_img_paths = search_img_paths(docus_img_paths)

        # paths to images
        self.docus_imgs = np.array(paths2imgs(docus_img_paths))
        self.stamp_imgs = np.array(paths2imgs(stamp_img_paths))

        # 도장의 개 수가 class 의 개수가 됩니다.
        self.n_classes = len(self.stamp_imgs) + 1

        # 도장 index을 설정합니다.
        self.stamp_indx = np.arange(len(self.stamp_imgs))

    @staticmethod
    def random_attach_stamp(doc, mask, stamp, coord_range, inplace):
        """
        문서와 문서와 같은 크기인 zero matrix의 지정된 위치에 도장을 찍고 찍힌 문서와 zero matrix을 반환합니다.

        :param doc: ndarray, 3d or 2d
        :param stamp: ndarray, 3d or 2d
        :param coord_range: (x y x y), 도장이 찍힐 문서 위치
        :param inplace: bool, doc 에 대한 inplace 여부를 결정합니다.
        :return: ndarray, 3d or 2d
        """

        if not inplace:
            doc = copy(doc)

        # stamp 의 크기
        stamp_w = stamp.shape[1]
        stamp_h = stamp.shape[0]

        # stamp 가 찍힐 documentation 상 random 한 위치 추출
        x1, y1, x2, y2 = coord_range
        gap_w = x1 + x2 - stamp_w
        gap_h = y1 + y2 - stamp_h
        rand_x = np.random.randint(x1, gap_w)
        rand_y = np.random.randint(y1, gap_h)

        # 빈 화면에 도장을 찍습니다. 해당 데이터는 segmentation 정답 데이터로 사용합니다.
        masked_doc = StampSignSegmentDataprovider.attatch_stamp(mask, stamp, coord=(rand_x, rand_y))

        # 이미지에 도장을 찍습니다.
        doc = StampSignSegmentDataprovider.attatch_stamp(doc, stamp, coord=(rand_x, rand_y))

        # show_image(patch_stamp)
        return doc, masked_doc, (rand_x, rand_y, rand_x + stamp_w, rand_y + stamp_h)

    @staticmethod
    def attatch_stamp(doc, stamp, coord):
        """
        doc 에 지정된 위치에 stamp 을 찍고 doc 을 반환합니다(inplace 방법)

        :param doc: 도장이 찍힐 문서, np.array, shape=(h, w, ch)
        :param stamp: np.array, shape=(h, w, ch)
        :param coord: 도장이 찍힐 좌측 상단의 위치
        :return: doc: 도장이 찍힌 문서, np.array, shape=(h, w, ch)
        """
        x1, y1 = coord
        stamp_w = stamp.shape[1]
        stamp_h = stamp.shape[0]

        patch_stamp = np.where(stamp != 0, stamp, doc[y1: y1 + stamp_h, x1: x1 + stamp_w])
        doc[y1: y1 + stamp_h, x1: x1 + stamp_w] = patch_stamp
        return doc

    def __len__(self):
        # 1 epochs 당 step 수를 결정합니다.
        return len(self.docus_imgs)

    def __getitem__(self, index):
        """
        Description:
            하나의 문서에 여러 종류의 도장을 찍습니다. 도장의 개 수는 하나의 문서에 약 0개 부터 10개 미만 입니다.
            한 step 에서 하나의 document 만 사용됩니다.

            xs, ys 값을 출력합니다.
            xs 는 이미지, ys는 localization(delta) 값과 class 정보가 concatenate 된 형태로 제공됩니다.
            class 정보는 onehot flag 여부에 따라 제공 여부가 결정 됩니다.
            examples)
                (Δcx,Δcy,Δw,Δh, class) or (Δcx,Δcy,Δw,Δh, 0, 0, 1, 0...)
        Args:
            index: int, batch index, 몇 번째 batch 묶음이 전닫될지 결정합니다.
        Returns:
            batch_xs: ndarary, (N_data, image height , image W, image CH)
            batch_ys: ndarray, (N_data, N_anchor, 4+1) or (N_data, N_anchor, 4+n_classes)

        """

        # 하나의 document 을 가져옵니다.
        doc = self.docus_imgs[index]
        batch_docs = copy_obj(doc, self.batch_size)

        # random 한 위치에 도장이 찍힌 documentation 을 반환합니다.
        batch_xs = []
        batch_ys = []
        self.n_stamps = []
        for ind, docu in enumerate(batch_docs):

            # doc 내 stamp 을 몇 번 찍을지 지정합니다.
            n_stamp = np.random.randint(np.min(self.n_stamp_range), np.max(self.n_stamp_range) + 1)
            self.n_stamps.append(n_stamp)

            # doc 내 도장을 하나 이상 찍는걸 보증합니다.
            assert n_stamp > 0

            # 도장 pool에서 random 으로 지정된 개 수(n_stamp) 만큼 도장을 추출합니다.
            trgt_cls = np.random.choice(self.stamp_indx, size=n_stamp)
            trgt_stamps = self.stamp_imgs[trgt_cls]

            # 문서 전체에서 stamp 가 찍히도록 설정 합니다.
            stamp_loc = [0, 0, docu.shape[1], docu.shape[0]]

            # 선택된 도장을 문서와 zero-matirx 찍습니다.
            stamp_coords = []
            mask = np.zeros_like(docu)
            for trgt_stamp in trgt_stamps:
                # 지정된 범위 내 random 한 위치에 도장이 찍히도록 합니다.
                docu, mask, coord = StampSignSegmentDataprovider.random_attach_stamp(docu, mask, trgt_stamp, stamp_loc,
                                                                                     inplace=False)
                stamp_coords.append(coord)

            # generate boolean mask
            mask = mask.sum(axis=-1)
            mask = np.where(mask == 0, 0, 1)

            # random offset 생성
            stamp_coords = np.array(stamp_coords)
            offset_xs, offset_ys = random_offset(stamp_coords, (0.25, 0.25))

            # shape (N_stamp)  => (N_stamp, 1)
            offset_xs = np.expand_dims(offset_xs, axis=-1)
            offset_ys = np.expand_dims(offset_ys, axis=-1)

            # random offset 적용
            # shape : (N_stamp, 2) += (N_stamp, 1) => (N_stamp, 2)
            stamp_coords[:, [0, 2]] += offset_xs
            stamp_coords[:, [1, 3]] += offset_ys

            stamp_coords = np.array(stamp_coords)
            batch_xs.append(crop_images(docu, stamp_coords))
            batch_ys.append(crop_images(mask, stamp_coords))

        return batch_xs, batch_ys


if __name__ == '__main__':
    docs_folder = '../dataset/docs_preproc'
    stamp_folder = '../dataset/stamp_vector'

    sss_dp = StampSignSegmentDataprovider(stamp_folder, docs_folder, 32)
    batch_xs, batch_ys = sss_dp[0]
    plot_images(batch_ys[0])
