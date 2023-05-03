import glob
import os
import shutil
import sys

import geopandas as gpd
import numpy as np
from eodag import setup_logging
from eodag.api.core import EODataAccessGateway
from eoreader.bands import VV, VV_DSPK, VH, VH_DSPK
from eoreader.reader import Reader
from osgeo import gdal, osr
from sentinelhub.geo_utils import wgs84_to_utm, to_wgs84, get_utm_crs
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import trange

workspace = 'C:/diploma/sip-service-main/sip-service-auth/src/main/python/images/source'
setup_logging(3)
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = "katarina.spasenovic@omni-energy.it"
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = "M@rkon!1997"

# os.environ["EODAG__ONDA__AUTH__CREDENTIALS__USERNAME"] = ["t0pcup@yandex.ru", "kdmikhaylova_1@edu.hse.ru"][0]
# os.environ["EODAG__ONDA__AUTH__CREDENTIALS__PASSWORD"] = ["jL7-iq4-GBM-RPe", "b8k-Jyy-NzS-jZ6"][0]

shutil.rmtree(workspace)
if not os.path.isdir(workspace):
    os.mkdir(workspace)

yaml_content = """
peps:
    download:
        outputs_prefix: "{}"
        extract: true
""".format(workspace)

with open(f'{workspace}/eodag_conf.yml', "w") as f_yml:
    f_yml.write(yaml_content.strip())

dag = EODataAccessGateway(f'{workspace}/eodag_conf.yml')
product_type = 'S1_SAR_GRD'
dag.set_preferred_provider("peps")
save_path = 'C:/diploma/sip-service-main/sip-service-auth/src/main/python/images/input'

for i in os.listdir(save_path):
    os.remove(f'{save_path}/{i}')
url = "http://services.sentinel-hub.com/ogc/wms/b351739d-40a8-4e8a-b943-701ef8249e08?SERVICE=WMS&REQUEST=GetMap&CRS" \
      "=EPSG:3857&SHOWLOGO=false&VERSION=1.3.0&LAYERS=IW_VV_DB&MAXCC=1&WIDTH=256&HEIGHT=256&FORMAT=image/jpeg&TIME" \
      "=2022-02-01/2022-04-10&GEOMETRY=POLYGON((-15220335.352339389 10695022.328252906,-15136996.786161486 " \
      "10695022.328252906,-15136996.786161486 10768110.696451709,-15220335.352339389 10768110.696451709," \
      "-15220335.352339389 10695022.328252906))"
url = url.replace('jpeg', 'tiff')
order_id = '04bd8753-5d42-49a9-a64b-e539df0a8ada'
# order_id = sys.argv[1]
# url = sys.argv[2].replace('jpeg', 'tiff')

url_crs = url.split('&SHOWLOGO')[0].split('CRS=')[1]
date1, date2 = url.split('&TIME=')[1].split('&GEOMETRY=')[0].split('/')
url_poly = url.split("GEOMETRY=")[1]

dataset = gpd.GeoSeries.from_wkt(data=[url_poly], crs=url_crs).to_crs('epsg:4326')
search_criteria = {
    "productType": product_type,
    "start": f'{date1}T00:00:00',
    "end": f'{date2}T23:59:59',
    "geom": Polygon(dataset.iloc[0]),
    "items_per_page": 500,
}

first_page, estimated = dag.search(**search_criteria)
if estimated == 0:
    sys.exit("no estimated")

print('download started, estimated=', estimated)
for item in first_page:
    if {'1SDH', 'EW'} & set(item.properties["title"].split('_')):
        continue
    try:
        product_path = item.download(extract=False)
        print(item)
        break
    except:
        pass
print('download finished')

zip_paths = [glob.glob(f'{workspace}/*1SDV*.zip')[0]]
for zip_id in trange(len(zip_paths), ascii=True):
    if not os.path.isfile(zip_paths[zip_id]):
        continue
    full_path = os.path.join(workspace, zip_paths[zip_id])
    reader = Reader()
    try:
        product = reader.open(full_path)
    except:
        continue

    product_poly = product.wgs84_extent().iloc[0].geometry
    name = os.path.basename(full_path)
    crs = get_utm_crs(product_poly.bounds[0], product_poly.bounds[1])

    dataset = gpd.GeoSeries.from_wkt(data=[url_poly], crs=url_crs).to_crs('epsg:4326')
    poly = Polygon(unary_union(dataset.iloc[0]))
    inter_area = product_poly.intersection(poly).area
    if inter_area == 0:
        continue

    bands = [VV, VV_DSPK, VH, VH_DSPK]
    ok_bands = [band for band in bands if product.has_band(band)]
    if len(ok_bands) != 4:
        continue

    stack = product.stack(ok_bands)
    np_stack = stack.to_numpy()
    resolution = product.resolution
    chunk_size = int((256 / 20) * 100)

    min_utm = wgs84_to_utm(poly.bounds[0], poly.bounds[1], crs)
    max_utm = wgs84_to_utm(poly.bounds[2], poly.bounds[3], crs)
    min_utm = max(min_utm[0], stack.x[0]), max(min_utm[1], stack.y[-1])
    max_utm = min(max_utm[0], stack.x[-1]), min(max_utm[1], stack.y[0])

    if min_utm[0] > max_utm[0] or min_utm[1] > max_utm[1]:
        continue

    min_x = int((np.abs(stack.x - min_utm[0])).argmin())
    max_y = int((np.abs(stack.y - min_utm[1])).argmin())
    max_x = int((np.abs(stack.x - max_utm[0])).argmin())
    min_y = int((np.abs(stack.y - max_utm[1])).argmin())

    step_x = (max_x - min_x) // chunk_size
    step_y = (max_y - min_y) // chunk_size
    for sx in range(step_x + 1):
        for sy in range(step_y + 1):
            try:
                tiff_name = f'{save_path}/{order_id}_{sx}_{sy}'
                if tiff_name in os.listdir(save_path):
                    continue

                y1 = min_y + sy * chunk_size
                y2 = min_y + (sy + 1) * chunk_size

                x1 = min_x + sx * chunk_size
                x2 = min_x + (sx + 1) * chunk_size
                if sum([y1 < 0, x1 < 0, y2 >= len(stack.y), x2 >= len(stack.x)]):
                    continue

                patch = np_stack[:, y1:y2, x1:x2]
                if np.sum(np.isnan(patch)) > 20 * chunk_size * 4:
                    continue

                x_min, y_max = to_wgs84(stack.x[x1], stack.y[y1], crs)
                x_max, y_min = to_wgs84(stack.x[x2], stack.y[y2], crs)
                nx, ny = patch.shape[1], patch.shape[1]

                x_res, y_res = (x_max - x_min) / float(nx), (y_max - y_min) / float(ny)
                geo_transform = (x_min, x_res, 0, y_max, 0, -y_res)

                np.save(tiff_name, patch)

                bd = 2 if len(ok_bands) == 2 else 3
                dst_ds = gdal.GetDriverByName('GTiff').Create(f'{tiff_name}.tiff', ny, nx, bd, gdal.GDT_Byte)
                dst_ds.SetGeoTransform(geo_transform)  # specify coords
                srs = osr.SpatialReference()  # establish encoding
                srs.ImportFromEPSG(4326)  # WGS84 lat/long
                dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
                dst_ds.GetRasterBand(1).WriteArray(patch[0])  # write r-band to the raster
                dst_ds.GetRasterBand(2).WriteArray(patch[1])  # write g-band to the raster
                if bd == 3:
                    dst_ds.GetRasterBand(3).WriteArray(patch[2])  # write b-band to the raster
                dst_ds.FlushCache()
            except:
                print(f'FAIL at {sx}-{sy}', end='')
                pass
