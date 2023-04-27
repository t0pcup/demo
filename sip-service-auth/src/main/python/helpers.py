import os
import torch
import shapely
import datetime
import rasterio
import requests
import psycopg2
import warnings
import PIL.Image

import numpy as np
import torch.nn as nn
import geopandas as gpd
import albumentations as A
import segmentation_models_pytorch as smp
import torchvision

from rasterio.features import shapes
from shapely import wkt, symmetric_difference
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.losses import DiceLoss

# For ice_sent-2:
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# from tensorflow.keras.optimizers import legacy


# -------------------HELPERS-HELPERS-HELPERS-HELPERS-HELPERS-HELPERS-HELPERS-HELPERS-HELPERS-HELPERS------------------

def load_and_save_image(url, path):
    x = requests.get(url)

    with open(path, 'wb') as f:
        f.write(x.content)

    return x.content


def process_image(model_path, input_img_path, output_img_path):
    sat_img = rasterio.open(input_img_path, 'r')
    profile = sat_img.profile
    profile['count'] = 1
    transform = A.Compose([A.Resize(256, 256, p=1.0, interpolation=3)])

    img = rasterio.open(input_img_path, 'r').read()
    img = np.asarray(img)[:3].transpose((1, 2, 0))
    abc = PIL.Image.fromarray(img)
    img = transform(image=img)['image']
    img = np.transpose(img, (2, 0, 1))
    img = img / 255.0
    img = torch.tensor(img)

    model = SegmentationModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    logits_mask = model(img.to('cpu', dtype=torch.float32).unsqueeze(0))
    pred_mask = torch.sigmoid(logits_mask)
    pred_mask = (pred_mask.squeeze(0) > 0.6) * 1.0

    # Save output image:
    with rasterio.open(output_img_path, 'w', **profile) as src:
        src.write(pred_mask)


def process_image_ice(model_path, img, output_img_path, img_vv):
    transform = A.Compose([A.Resize(256, 256, p=1.0, interpolation=3)])

    img = np.asarray(img).transpose((1, 2, 0))
    img = transform(image=img)['image']
    img = np.transpose(img, (2, 0, 1))

    img = stand(img, single_stand=True)

    img = new_normalize(img, plural=True)
    to_pil = (img[:3].transpose((1, 2, 0)) * 255).astype(np.uint8)
    # PIL.Image.fromarray(to_pil).show()

    model = smp.DeepLabV3(
        encoder_name="timm-mobilenetv3_small_075",
        encoder_weights=None,
        in_channels=4,
        classes=6,
    ).to('cpu', dtype=torch.float32)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict']
    model.load_state_dict(state_dict)

    dataset = InferDataset(output_img_path.split("outputs")[0] + "inputs",
                           os.listdir(output_img_path.split("outputs")[0] + "inputs")[0])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=coll_fn, shuffle=False)

    inputs = next(iter(dataloader))
    inputs = inputs.to('cpu')
    model.eval()
    outputs = model(inputs)
    y_pred = np.array(torch.argmax(outputs, dim=1).cpu())

    y_pred[y_pred < 2] = 0.0
    # y_pred[y_pred == 5] = 0
    y_pred[y_pred != 0] = 1.0
    pred_to_pil = y_pred[0].astype(np.uint8)
    # PIL.Image.fromarray(pred_to_pil).show()

    profile = img_vv.profile
    profile["count"] = 1
    with rasterio.open(output_img_path, 'w', **profile) as src:
        src.write(pred_to_pil, 1)
    return y_pred[0]


# ---------VECTORIZE-VECTORIZE-VECTORIZE-VECTORIZE-VECTORIZE-VECTORIZE-VECTORIZE-VECTORIZE-VECTORIZE-VECTORIZE--------

def rasterio_geo(bytestream, mask):
    geom_all = []
    with rasterio.open(bytestream) as dataset:
        for geom, val in shapes(mask, transform=dataset.transform):
            if val != 0.0:
                geom_all.append(geom)  # , precision=6
                # geom_all.append(rasterio.warp.transform_geom(dataset.crs, 'EPSG:3857', geom))  # , precision=6
        return geom_all


def t(lst: list):
    return shapely.Polygon([(k[0], k[1]) for k in [j for j in lst[0]]])


def save_gj(url, path, p, c):
    sp = path.replace('p/', '')
    a = gpd.GeoSeries(p).to_json()

    p2 = wkt.loads(url.split('GEOMETRY=')[1])
    b = gpd.GeoSeries(p2).to_json()
    result = b.split("bbox\": [")[1].split("]")[0]

    gpd.GeoSeries(p).to_file(f"{sp}{c}.json", driver='GeoJSON', show_bbox=False)  # , crs="EPSG:4326"
    return a, result
# def save_gj(url, path, p, c): TODO: 111
#     sp = path.replace('p/', '')
#
#     p2 = wkt.loads(url.split('GEOMETRY=')[1])
#     b = gpd.GeoSeries(p2).to_json()
#     result = b.split("bbox\": [")[1].split("]")[0]
#
#     with open(f"{sp}{c}.json", "w") as json_f:
#         json_f.write(p)
#     # p.to_file(f"{sp}{c}.json", driver='GeoJSON', show_bbox=False)  # , crs="EPSG:4326"
#     return p, result


def to_pol(j):
    return t(j['coordinates'])


def RasterioGeo(bytestream, mask, key):
    geom_all = []
    with rasterio.open(bytestream) as dataset:
        for geom, val in shapes(mask, transform=dataset.transform):
            if val == key:
                # g_w = rasterio.warp.transform_geom('EPSG:3857', 'EPSG:3857', geom)  # , precision=6
                geom_all.append(geom)
        return geom_all


def vectorize(url, output_img_path):
    path = ''
    warnings.filterwarnings("ignore")
    colors = {
        'white': 1.0,
    }
    for image_name in [output_img_path]:
        image = rasterio.open(path + image_name, 'r')
        im_arr = np.asarray(image.read()).transpose((1, 2, 0))

        masks = [im_arr.reshape(im_arr.shape[:2])]
        for i in range(len(colors)):
            mask_2d = masks[i]

            h, w = mask_2d.shape
            mask_3d = np.zeros((h, w), dtype='uint8')
            mask_3d[mask_2d[:, :] > 0.5] = 255
            if np.sum(mask_3d) == 0:
                print('no such colour: ' + list(colors.keys())[i])
                continue

            # Image.fromarray(np.uint8(mask_3d)).show()

            polygons = [to_pol(p) for p in rasterio_geo(path + image_name, mask_3d)]
            if len(polygons) == 0:
                print('no suitable polygons left')
                continue

            mp = shapely.MultiPolygon(polygons)
            return *save_gj(url, path, mp, image_name.split('.')[0] + '_' + list(colors.keys())[i]), mp
# def vectorize(url, output_img_path, y_p): TODO: 111
#     path = ''
#     warnings.filterwarnings("ignore")
#
#     for image_name in [output_img_path]:
#         image = rasterio.open(path + image_name, 'r')
#         im_arr = np.asarray(image.read()).transpose((1, 2, 0))
#
#         masks = []
#         mask_2d = im_arr.reshape(im_arr.shape[:2])
#         str_res = []
#         mps = []
#         keys = ''
#         for key in np.unique((np.asarray(1 + y_p[:, :])).astype(np.uint8)):
#             polygons = [to_pol(p) for p in
#                         RasterioGeo(path + image_name, (np.asarray(1 + y_p[:, :])).astype(np.uint8), key)]
#
#             mp = shapely.geometry.MultiPolygon(polygons[:-1])
#             str_res.append(str(key) + ' ' + gpd.GeoSeries([mp]).to_json())
#             mps.append(mp)
#             keys = keys + str(key) + ';'
#             if len(polygons) == 0:
#                 print('no suitable polygons left')
#                 continue
#
#         return *save_gj(url, path, '\n'.join(str_res),
#                         image_name.split('.')[0] + '_' + keys), mps


# -----------DATABASE-DATABASE-DATABASE-DATABASE-DATABASE-DATABASE-DATABASE-DATABASE-DATABASE-DATABASE----------

def save_order_result_to_database(db, order):
    try:
        connection = psycopg2.connect(
            user=db["user"],
            password=db["password"],
            host=db["host"],
            port=db["port"],
            database=db["database"]
        )
        cursor = connection.cursor()

        q = """UPDATE orders
                SET status = %s, url = %s, url2 = %s, finished_at = %s, result = %s, result2 = %s, bbox = %s, diff = %s
                WHERE id = %s"""
        record = (
            order["status"],
            order["url"],
            order["url2"],
            datetime.datetime.now(),
            order["result"],
            order["result2"],
            order["bbox"],
            order["diff"],
            order["order_id"],
        )
        cursor.execute(q, record)
        connection.commit()

    except (Exception, psycopg2.Error) as error:
        print("Failed to insert record into mobile table", error)

    finally:
        # closing database connection.
        if connection is not None and connection:
            cursor.close()
            connection.close()


# -----------------------MODELS-MODELS-MODELS-MODELS-MODELS-MODELS-MODELS-MODELS-MODELS-MODELS----------------------

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()

        self.arc = smp.Unet(
            encoder_name='resnet50',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, images, masks=None):
        logits = self.arc(images)

        if masks is not None:
            loss1 = DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1, loss2
        return logits


# -----------------------ICE_SENT1--ICE_SENT1--ICE_SENT1--ICE_SENT1--ICE_SENT1--ICE_SENT1--------------------------

global profile


def save_t(full_name, im, prof):
    with rasterio.open(full_name, 'w', **prof) as src:
        src.write(im)


def item_getter(path: str, file_name: str, transforms=A.Compose([
    A.CenterCrop(256, 256, p=1.0, always_apply=False)]), val=False):
    image = rasterio.open(path + '/input_ice.tiff', 'r').read()

    img = image.transpose((1, 2, 0))
    augmented = transforms(image=img)
    image = A.Compose([A.Resize(256, 256, p=1.0, interpolation=3)])(image=augmented['image'])['image']
    image = image.transpose((2, 0, 1))
    assert not np.any(np.isnan(image))
    return image


def coll_fn(batch_):
    ims_, labels_ = [], []
    for _, sample in enumerate(batch_):
        im = sample
        ims_.append(torch.from_numpy(im.copy()))
    return torch.stack(ims_, 0).type(torch.FloatTensor)


def new_normalize(im_: np.ndarray, single_norm=False, plural=False) -> np.ndarray:
    im_ = np.nan_to_num(im_)

    mean, std = np.zeros(im_.shape[0]), np.zeros(im_.shape[0])
    for channel in range(im_.shape[0]):
        mean[channel] = np.mean(im_[channel, :, :])
        std[channel] = np.std(im_[channel, :, :])
    print(mean)
    print(std)
    norm = torchvision.transforms.Normalize(mean, std)
    return np.asarray(norm.forward(torch.from_numpy(im_)))


def stand(im_: np.ndarray, single_stand=False) -> np.ndarray:
    # TODO: try *(16.50119 + 49.44221) - (16.50119 + 49.44221)
    im_ = np.nan_to_num(im_) / 255.0
    min_ = np.array([-49.44221, -49.44221, -49.679745, -49.679745])
    max_ = np.array([16.50119, 15.677849, 2.95751, 2.9114623])

    print(min_)
    print(max_)
    for channel in [0, 1]:
        im_[channel] = im_[channel] * (max_[channel] - min_[channel]) + min_[channel]
        print(channel, np.min(im_[channel]), np.max(im_[channel]))
    for channel in [2, 3]:
        im_[channel] = im_[channel] * (max_[channel] - min_[channel]) + min_[channel]
        print(channel, np.min(im_[channel]), np.max(im_[channel]))

    # if single_stand:
    #     min_, max_ = np.zeros(im_.shape[0]), np.zeros(im_.shape[0])
    #     for channel in range(im_.shape[0]):
    #         min_[channel] = np.min(im_[channel, :, :])
    #         max_[channel] = np.max(im_[channel, :, :])

    # for channel in [0, 1]:
    #     im_[channel] = (im_[channel] - min_[channel]) / (max_[channel] - min_[channel])
    #     im_[channel] = (im_[channel] * (16.50119 + 49.44221 - 42.3)) - (16.50119 + 49.44221) + 41.18
    # for channel in [2, 3]:
    #     im_[channel] = (im_[channel] - min_[channel]) / (max_[channel] - min_[channel])
    #     im_[channel] = (im_[channel] * (2.95751 + 49.679745 - 15.0)) - (2.95751 + 49.679745) + 18.0
    return im_
    # return (im_ * 20.0) - 20.0


class InferDataset(Dataset):
    def __init__(self, path, file_n):
        self.path = path
        self.data_list = [file_n]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        f_name = self.data_list[index]
        image = item_getter(self.path, f_name, val=False)
        return image


# -----------------------DIFFERENCE-DIFFERENCE-DIFFERENCE-DIFFERENCE-DIFFERENCE-DIFFERENCE--------------------------

def get_symmetric_difference(mp1, mp2):
    diff = symmetric_difference(mp1, mp2)
    gpd.GeoSeries(diff).simplify(tolerance=500).to_file(f"diff.json", driver='GeoJSON', show_bbox=False)
    return gpd.GeoSeries(diff).to_json()


# -----------------------ICE_SENT2--ICE_SENT2--ICE_SENT2--ICE_SENT2--ICE_SENT2--ICE_SENT2--------------------------

# profile = None


def read_image(img_path):
    # image = tf.io.read_file(img_path)
    # image = tf.image.decode_jpeg(image, channels=3)
    image = rasterio.open("images/inputs/a.tiff", 'r').read()
    image = tf.convert_to_tensor(image[:3].transpose((1, 2, 0)), np.float32)
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0

    return image


# def display(display_list, n_colors):
#     ice_colors = n_colors - 1
#     new_colors = plt.get_cmap('jet', ice_colors)(np.linspace(0, 1, ice_colors))
#     black = np.array([[0, 0, 0, 1]])
#     # white = np.array([[1, 1, 1, 1]])
#     new_colors = np.concatenate((new_colors, black), axis=0)  # land will be black
#     cmap = ListedColormap(new_colors)
#
#     fig, axs = plt.subplots(nrows=1, ncols=len(display_list), figsize=(15, 6))
#
#     title = ['Input Image', 'Predicted Mask']
#
#     pred_mask = None
#     for i in range(len(display_list)):
#         axs[i].set_title(title[i])
#         if i == 0:
#             _ = 0
#             # axs[i].imshow(tf.keras.utils.array_to_img(display_list[i]))
#         else:
#             pred_mask = display_list[i]
#             pred_mask_numpy = pred_mask.numpy()
#
#             pred_mask_numpy[pred_mask_numpy > 6] = 0
#             pred_mask_numpy[pred_mask_numpy < 4] = 0
#             pred_mask_numpy[pred_mask_numpy != 0] = 1
#             pred_to_pil = pred_mask_numpy.astype(np.uint8)
#             im = PIL.Image.new(mode='1', size=pred_to_pil.shape[:2])
#             im.putdata(pred_to_pil[:, :, 0].flatten())
#             # im.show()
#
#             global profile
#
#             profile["count"] = 1
#             with rasterio.open("result.tiff", 'w', **profile) as src:
#                 src.write(np.array(im), 1)
#
#             # msk = axs[i].imshow(display_list[i], cmap=cmap, vmin=0, vmax=n_colors - 1)
#         axs[i].axis('off')
#
#     cbar = fig.colorbar(msk, ax=axs, location='right')
#     tick_locs = (np.arange(n_colors) + 0.5) * (n_colors - 1) / n_colors
#     cbar.set_ticks(tick_locs)
#     cbar.set_ticklabels(np.arange(n_colors))
#     plt.savefig('ice2_result.png')
#     plt.show()
#
#     return pred_mask


def create_mask(pred_mask, ele=0):
    pred_mask = tf.argmax(pred_mask, axis=-1)  # use the highest proabbaility class as the prediction
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[ele]


# def show_predictions(model, dataset, num=1, ele=0, n_colors=8):
#     global profile
#     for image in dataset.take(num):
#         profile = image.profile
#         p = model.predict(image)
#         a = create_mask(p, ele)
#         display([image[ele], a], n_colors=8)


class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
                 y_true=None,
                 y_pred=None,
                 num_classes=None,
                 name=None,
                 dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

# def run_ice2(model_path, img_path, n_colors=8):
#     model = tf.keras.models.load_model(model_path, compile=False)
#     model.compile(
#         optimizer=legacy.Adam(),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#         metrics=['sparse_categorical_accuracy', UpdatedMeanIoU(num_classes=n_colors)]
#     )
#
#     val_dataset = tf.data.Dataset.from_tensor_slices(([img_path])).map(read_image).batch(1)
#     return show_predictions(model, val_dataset, num=10, ele=0, n_colors=n_colors)

# run_ice2("models/saved_model/ice", "images/inputs/a.tiff")
