from tensorflow.keras.models import Model
from dataprovider import StampSignSegmentDataprovider

from unet import simple_unet

if __name__ == '__main__':
    # dataload
    docs_folder = '../dataset/docs_preproc'
    stamp_folder = '../dataset/stamp_vector'
    sss_dp = StampSignSegmentDataprovider(stamp_folder, docs_folder, (112, 112), 64)

    inputs, pred = simple_unet((112, 112, 3), sss_dp.n_classes)
    model = Model(inputs, pred)
    model.compile('adam', 'mse', 'acc')
    model.fit(sss_dp)
