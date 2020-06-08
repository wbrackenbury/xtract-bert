import io
import tensorflow as tf
import numpy as np
from PIL import Image
import logging
import argparse

RESNET_SIZE = (224, 224)

def imbytes_to_imformat(fb):

    """
    Takes bytes from an image and turns it into a representation
    we can use for classification
    """

    image = Image.open(io.BytesIO(fb))
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    if img_arr.shape[-1] == 1:
        img_arr = tf.tile(img_arr, [1, 1, 3])
    elif img_arr.shape[-1] == 4:
        img_arr = img_arr[:, :, :3]
    img_arr = tf.image.resize(img_arr[tf.newaxis, :, :, :], RESNET_SIZE)

    return img_arr

def conv_resnet_labels(pred_obj):

    """
    Prediction object looks like:

[
   [
      ('n07753592', 'banana', 0.99229723),
      ('n03532672', 'hook', 0.0014551596),
      ('n03970156', 'plunger', 0.0010738898),
      ('n07753113', 'fig', 0.0009359837) ,
      ('n03109150', 'corkscrew', 0.00028538404)
   ]
]

    And we want to get the labels from each.

    """

    #print(pred_obj)

    decoded_pred = tf.keras.applications.imagenet_utils.decode_predictions(pred_obj)
    unbatch = decoded_pred[0]
    get_pred_obj = lambda x: x[1]
    labels = [get_pred_obj(o) for o in unbatch]

    return labels

def conv_to_web_labels(labels):

    """
    We're stuck with a rough legacy format--web labels are formatted
    in the database as a string

    '[(label, None)]'

    So we have to convert to that from the labels

    """

    return [(l, None) for l in labels]

def get_im_model():

    resnet = tf.keras.applications.resnet50.ResNet50(weights='imagenet')

    print(resnet.summary())

    out_layer = resnet.get_layer('avg_pool')
    identity = tf.keras.layers.Lambda(lambda x: x)(out_layer.output)
    pred_layer = resnet.get_layer('probs')(out_layer.output)

    model = tf.keras.models.Model(inputs = resnet.input,
                                  outputs = [identity, pred_layer])

    for l in model.layers:
        l.trainable = False

    return model

def get_fb(fname):

    with open(fname, 'rb') as of:
        fb = of.read()
    return fb

def finalize_im_rep(fname):

    fb = get_fb(fname)
    model = get_im_model()

    try:
        im = imbytes_to_imformat(fb)
        im_rep, label_preds = model.predict(im)
        full_labels = conv_resnet_labels(label_preds)
        return im_rep[0], full_labels

    except Exception as e:
        logging.error(e)
        return None, None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to directory',
                        required=True, type=str)
    args = parser.parse_args()

    rep, labels = finalize_im_rep(args.path)
    print(rep, labels)
