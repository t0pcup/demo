import sys

from helpers import *

root_path = "C:/diploma/sip-service-main/sip-service-auth/src/main/python/"
# root_path = ""  # for compiling with python configurer

# Get all arguments:
order_id = sys.argv[1]
url = sys.argv[2].replace('jpeg', 'tiff')

model_name = "ice.pth"
# url2 = sys.argv[3].replace('jpeg', 'tiff')
# model_name = models[sys.argv[4]]

# order_id = "06ef1278-4b9b-494f-b8b8-1e59f5b5bacf"
# url = "http://services.sentinel-hub.com/ogc/wms/b351739d-40a8-4e8a-b943-701ef8249e08?SERVICE=WMS&REQUEST=GetMap&CRS" \
#       "=EPSG:3857&SHOWLOGO=false&VERSION=1.3.0&LAYERS=IW_VV_DB&MAXCC=1&WIDTH=256&HEIGHT=256&CRS=EPSG:3857&FORMAT" \
#       "=image/jpeg&TIME=2023-04-04/2023-04-07&GEOMETRY=POLYGON((-13426722.855917903 9734912.122948073," \
#       "-13350429.787929332 9734912.122948073,-13350429.787929332 9800563.057195181,-13426722.855917903 " \
#       "9800563.057195181,-13426722.855917903 9734912.122948073))"
# url = url.replace('jpeg', 'tiff')
# Set paths:
model_path = f'{root_path}models/{model_name}'
input_img_path = f'{root_path}images/inputs/input_{model_name.split(".")[0]}.tiff'
output_img_path = f'{root_path}images/outputs/output_{model_name.split(".")[0]}.tiff'

result, result2, bbox, mp1, mp2 = None, None, None, None, None

if model_name == "ice.pth":
    in_img = model_name.split('.')[0]
    vv_img_path = f'{root_path}images/inputs/vv_{in_img}.tiff'
    vh_img_path = f'{root_path}images/inputs/vh_{in_img}.tiff'
    vv20_img_path = f'{root_path}images/inputs/vv20_{in_img}.tiff'
    vh20_img_path = f'{root_path}images/inputs/vh20_{in_img}.tiff'

    load_and_save_image(url.replace("IW_VV_DB", "IW_VV_DB"), vv_img_path)
    load_and_save_image(url.replace("IW_VV_DB", "IW-VH-DB"), vh_img_path)
    load_and_save_image(url.replace("IW_VV_DB", "IW_VV_DB"), vv20_img_path)
    load_and_save_image(url.replace("IW_VV_DB", "IW-VH-DB"), vh20_img_path)

    img_vv = rasterio.open(vv_img_path, 'r')
    img_vh = rasterio.open(vh_img_path, 'r')
    img_vv20 = rasterio.open(vv20_img_path, 'r')
    img_vh20 = rasterio.open(vh20_img_path, 'r')
    profile = img_vv.profile
    profile['count'] = 4

    img = np.ndarray(shape=(4, *img_vv.shape))
    img[0] = np.array(img_vv.read(1))
    img[1] = np.array(img_vv20.read(1))
    img[2] = np.array(img_vh.read(1))
    img[3] = np.array(img_vh20.read(1))

    with rasterio.open(input_img_path, 'w', **profile) as src:
        src.write(img)

    y_p = process_image_ice(model_path, img, output_img_path, img_vv)
    result, bbox, mp1 = vectorize(url, output_img_path)

diff = None
if result2 is not None:
    try:
        diff = get_symmetric_difference(mp1, mp2)
    except ():
        pass

# ----------------------------------------------------DATABASE----------------------------------------------------

# Update the finished order in the database:
db = {
    "host": "127.0.0.1",
    "port": "5432",
    "user": "postgres",
    "password": "20010608Kd",
    "database": "db",
}
order = {
    "status": "true",
    "url": url,
    "url2": None,
    "bbox": bbox,
    "result": result,
    "result2": result2,
    "order_id": order_id,
    "diff": diff,
}
save_order_result_to_database(db, order)
