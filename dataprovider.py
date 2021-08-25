import os
from copy import copy
from glob import glob

from tensorflow.keras.utils import Sequence
from tensorflow.python.keras.utils.np_utils import to_categorical

from utils import random_offset, crop_images, search_img_paths, paths2imgs, xyxy2xywh, copy_obj
import numpy as np


class StampSignSegmentDataprovider(Sequence):
    def __init__(self, stamp_folder, docs_folder, batch_size, input_shape, onehot=True, max_n_stamp=5):
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
        self.input_shape = input_shape
        self.onehot = onehot
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
    def random_attach_stamp(doc, stamp, coord_range, inplace):
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

        # 빈 화면에 도장을 찍습니다.
        masked_doc = StampSignSegmentDataprovider.attatch_stamp(doc, stamp, coord=(rand_x, rand_y))
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

            # doc 내 도장을 하나 이상 찍습니다.
            if n_stamp != 0:

                # 하나의 문서에 몇 개의 도장을 찍을지 결정합니다.
                trgt_cls = np.random.choice(self.stamp_indx, size=n_stamp)
                trgt_stamps = self.stamp_imgs[trgt_cls]

                # 문서 전체에서 stamp 가 찍히도록 설정 합니다.
                stamp_loc = [0, 0, docu.shape[1], docu.shape[0]]

                # 문서에 도장을 찍습니다.
                trgt_loc = []
                for trgt_stamp in trgt_stamps:
                    # 위 지정된 범위 내에 random 한 위치에 도장이 찍히도록 합니다.
                    _, mask, coord = self.attach_stamp(docu, trgt_stamp, stamp_loc, inplace=True)
                    trgt_loc.append(coord)
                trgt_loc = np.array(trgt_loc)
                trgt_loc = xyxy2xywh(trgt_loc)

                # 각 anchor 에 가장 적절한 goundtruth 와 stamp obj 간 delta 을 계산 및 cls 을 부여합니다.
                batch_xs, batch_ys = generate_segmentation_dataset(docu, trgt_loc, trgt_cls)

            # doc 내 도장을 하나도 찍지 않습니다.
            else:
                bg_delta = np.zeros_like(self.concat_default_boxes)
                bg_cls = np.ones(shape=(len(self.concat_default_boxes), 1)) * (self.n_classes - 1)
                delta_cls = np.concatenate([bg_delta, bg_cls], axis=-1)

        # change class to  onehot vector
        if self.onehot:
            batch_ys = np.array(batch_ys)
            batch_delta = batch_ys[:, :, :4]
            ys_onehot = to_categorical(batch_ys[:, :, 4], num_classes=self.n_classes)
            batch_ys = np.concatenate([batch_delta, ys_onehot], axis=-1)

        return np.array(batch_docs), np.array(batch_ys)
