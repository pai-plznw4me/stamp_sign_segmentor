from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model

from dataprovider import StampSignSegmentDataprovider


def simple_unet(shape, n_classes):

    inputs = Input(shape)
    n_classes = n_classes

    #  models
    # layer 1, Image shape (128, 128) -> (64, 64)
    layer = Conv2D(filters=16, kernel_size=5, padding='same', activation='relu', name='conv1_1',
                   kernel_initializer='he_normal')(inputs)
    layer = Conv2D(filters=16, kernel_size=5, padding='same', activation='relu', name='conv1_2',
                   kernel_initializer='he_normal')(layer)
    maxp1_3 = MaxPool2D(strides=2, padding='same', name='maxp1_3')(layer)

    # layer 2 Output Shape : (64, 64) -> (32, 32)
    layer = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv2_1',
                   kernel_initializer='he_normal')(maxp1_3)
    layer = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', name='conv2_2',
                   kernel_initializer='he_normal')(layer)
    maxp2_3 = MaxPool2D(strides=2, padding='same', name='maxp2_3')(layer)

    # layer 3 Output Shape : (32, 32) -> (16, 16)
    layer = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv3_1',
                   kernel_initializer='he_normal')(maxp2_3)
    layer = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv3_2',
                   kernel_initializer='he_normal')(layer)
    maxp3_3 = MaxPool2D(strides=2, padding='same', name='maxp3_3')(layer)

    # layer 4 Output Shape : (16, 16) -> (8, 8)
    layer = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name='conv4_1',
                   kernel_initializer='he_normal')(maxp3_3)
    layer = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', name='conv4_2',
                   kernel_initializer='he_normal')(layer)
    maxp4_3 = MaxPool2D(strides=2, padding='same', name='maxp4_3')(layer)

    # FC Layer to 1x1 Conv2D
    layer = Conv2D(512, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', activation='relu')(
        maxp4_3)
    layer = Conv2D(512, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal', activation='relu')(
        layer)
    layer = Conv2D(n_classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(layer)

    # Upsampling Layer
    up_activation = 'relu'
    up_layer = Conv2DTranspose(64, 3, 2, padding='same', activation=up_activation)(layer)
    up_layer = Conv2D(64, 3, 1, padding='same', activation=up_activation)(up_layer)
    concat_layer = Concatenate()([maxp3_3, up_layer])

    up_layer = Conv2D(32, 3, 1, padding='same', activation=up_activation)(concat_layer)
    up_layer = Conv2DTranspose(32, 3, 2, padding='same', activation=up_activation)(up_layer)
    concat_layer = Concatenate()([maxp2_3, up_layer])

    up_layer = Conv2D(16, 3, 1, padding='same', activation=up_activation)(concat_layer)
    up_layer = Conv2DTranspose(16, 3, 2, padding='same', activation=up_activation)(up_layer)
    concat_layer = Concatenate()([maxp1_3, up_layer])

    up_layer = Conv2D(16, 3, 1, padding='same', activation=up_activation)(concat_layer)
    up_layer = Conv2DTranspose(16, 3, 2, padding='same', activation=up_activation)(up_layer)
    pred = Conv2D(n_classes, 3, 1, padding='same', activation='softmax')(up_layer)

    return inputs, pred


if __name__ == '__main__':
    docs_folder = '../dataset/docs_preproc'
    stamp_folder = '../dataset/stamp_vector'
    sss_dp = StampSignSegmentDataprovider(stamp_folder, docs_folder, 2)
    xs, ys = sss_dp[0]

    inputs, pred = simple_unet((112, 112, 3))
    model = Model(inputs, pred)
    model.compile('adam', 'mse', 'acc')
    model.fit(sss_dp, epochs=60)
