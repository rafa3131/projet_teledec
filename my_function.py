import os
import numpy as np
from osgeo import gdal, ogr
import matplotlib.pyplot as plt
import libsigma.read_and_write as rw

gdal.UseExceptions()


# =====================================================
# UTILS
# =====================================================

def make_dir(base_dir, subdir):
    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


# =====================================================
# RASTERISATION EN MEMOIRE
# =====================================================

def rasterization_in_memory(
    in_vector,
    ref_image,
    field_name,
    dtype=gdal.GDT_Int16
):
    """
    Rasterise un shapefile selon la grille d'un raster
    et retourne le tableau numpy.
    """

    ref_ds = gdal.Open(ref_image)
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    xsize = ref_ds.RasterXSize
    ysize = ref_ds.RasterYSize

    mem_driver = gdal.GetDriverByName("MEM")
    out_ds = mem_driver.Create("", xsize, ysize, 1, dtype)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)

    band = out_ds.GetRasterBand(1)
    band.Fill(0)

    vect_ds = ogr.Open(in_vector)
    layer = vect_ds.GetLayer()

    gdal.RasterizeLayer(
        out_ds,
        [1],
        layer,
        options=[f"ATTRIBUTE={field_name}"]
    )

    array = band.ReadAsArray()

    vect_ds = None
    ref_ds = None
    out_ds = None

    return array


# =====================================================
# COMPTAGE PIXELS PAR STRATE (sans collections)
# =====================================================

def count_pixels_by_class(roi_array):
    roi_flat = roi_array.flatten()
    roi_flat = roi_flat[roi_flat > 0]
    classes = np.unique(roi_flat)
    counts = {c: np.sum(roi_flat == c) for c in classes}
    return counts


# =====================================================
# CALCUL ARI
# =====================================================

def calculate_ari(b03, b05):
    mask = (b03 == 0) | (b05 == 0)
    ari = (1/b03 - 1/b05) / (1/b03 + 1/b05)
    ari[mask] = np.nan
    return ari.astype(np.float32)


# =====================================================
# CALCUL STACK ARI
# =====================================================

def compute_ari_stack(b03_file, b05_file):
    arr_b03 = rw.load_img_as_array(b03_file).astype(np.float32)
    arr_b05 = rw.load_img_as_array(b05_file).astype(np.float32)

    n_dates = arr_b03.shape[2]  # dimensions: (y, x, band)
    ari_stack = []

    for i in range(n_dates):
        b03 = arr_b03[:, :, i]
        b05 = arr_b05[:, :, i]
        ari_stack.append(calculate_ari(b03, b05))

    return np.array(ari_stack)  # shape = (dates, y, x)


# =====================================================
# STATS PAR STRATE
# =====================================================

def stats_by_strate(ari_stack, roi_array, strate_ids):
    mean_dict = {}
    std_dict = {}

    for s in strate_ids:
        mask = roi_array == s
        if not np.any(mask):
            continue
        pixels = ari_stack[:, mask]  # shape = (dates, n_pixels)
        mean_dict[s] = np.nanmean(pixels, axis=1)
        std_dict[s] = np.nanstd(pixels, axis=1)

    return mean_dict, std_dict


# =====================================================
# GRAPHIQUE NB POLYGONES
# =====================================================

def plot_polygon_histogram(gdf, classe_col, classe_labels, out_file):
    counts = gdf[classe_col].value_counts().sort_index()
    classes = counts.index.tolist()
    nb_poly = counts.values
    labels = [f"{c} - {classe_labels[c]}" for c in classes]

    plt.figure(figsize=(6, 5))
    colors = ["#468fc0", "#ff8e2a", "#2ca02c"]
    bars = plt.bar(classes, nb_poly, color=colors)
    for bar, nb in zip(bars, nb_poly):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(nb_poly) * 0.01,
                 str(nb), ha="center", va="bottom")
    plt.xticks(classes, labels)
    plt.xlabel("Classe")
    plt.ylabel("Nombre de polygones")
    plt.title("Nombre de polygones par classe")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.show()


# =====================================================
# GRAPHIQUE NB PIXELS
# =====================================================

def plot_pixel_histogram(counts_pixel, classe_nom, out_file):
    classes = sorted(counts_pixel.keys())
    nb_pixel = [counts_pixel[c] for c in classes]
    labels = [f"{c} - {classe_nom[c]}" for c in classes]

    plt.figure(figsize=(6, 5))
    colors = ["#468fc0", "#ff8e2a", "#2ca02c"]
    bars = plt.bar(classes, nb_pixel, color=colors)
    for bar, nb in zip(bars, nb_pixel):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 str(nb), ha="center", va="bottom")
    plt.xticks(classes, labels)
    plt.xlabel("Classe")
    plt.ylabel("Nombre de pixels")
    plt.title("Nombre de pixels par classe")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.show()


# =====================================================
# TIF SERIE TEMPORELLE ARI
# =====================================================

def write_ari_timeseries_tif(ari_stack, ref_raster, out_tif, dates=None, nodata=-9999):
    # dimensions
    ref_ds = gdal.Open(ref_raster)
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()

    # convertir en 3D pour write_image
    arr_3d = np.transpose(ari_stack, (1, 2, 0)).copy()  # shape = (y, x, dates)
    arr_3d[np.isnan(arr_3d)] = nodata

    rw.write_image(
        out_tif,
        arr_3d,
        transform=gt,
        projection=proj,
        gdal_dtype=gdal.GDT_Float32,
        driver_name="GTiff"
    )


# =====================================================
# GRAPHIQUE SERIE ARI
# =====================================================

def plot_ari_timeseries(mean_dict, std_dict, strate_labels, dates, strate_ids, out_file):
    dates_plot = dates[::-1]
    x = np.arange(len(dates_plot))

    plt.figure(figsize=(9, 5))
    for s in strate_ids:
        if s not in mean_dict:
            continue
        plt.plot(x, mean_dict[s][::-1], label=f"{s} – {strate_labels[s]}")
        plt.fill_between(x,
                         mean_dict[s][::-1] - std_dict[s][::-1],
                         mean_dict[s][::-1] + std_dict[s][::-1],
                         alpha=0.25)
    plt.xticks(x, dates_plot, rotation=45)
    plt.xlabel("Date")
    plt.ylabel("ARI")
    plt.title("Série temporelle moyenne de l’ARI par strate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.show()
