import os

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, ogr

import libsigma.read_and_write as rw
from libsigma.classification import get_samples_from_roi
import libsigma.plots as plots

gdal.UseExceptions()


# ============================================================
# OUTILS DIVERS
# ============================================================

def make_dir(base_dir, subdir):
    """
    Crée (si besoin) puis renvoie un chemin de dossier: base_dir/subdir.
    """
    path = os.path.join(base_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def set_band_nodata(raster_path, nodata, band_index=1):
    """
    Fixe la valeur NoData d'un raster existant (en place).
    """
    ds = gdal.Open(raster_path, gdal.GA_Update)
    if ds is None:
        raise FileNotFoundError(raster_path)

    band = ds.GetRasterBand(band_index)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    ds = None


def set_band_descriptions(raster_path, descriptions):
    """
    Fixe les descriptions des bandes (ex: dates) d'un raster existant (en place).
    """
    ds = gdal.Open(raster_path, gdal.GA_Update)
    if ds is None:
        raise FileNotFoundError(raster_path)

    if ds.RasterCount != len(descriptions):
        raise ValueError("Nombre de bandes != nombre de descriptions.")

    for i, desc in enumerate(descriptions, start=1):
        ds.GetRasterBand(i).SetDescription(str(desc))

    ds = None


def _default_colors(n):
    """
    Renvoie une liste de couleurs (répétée si besoin) pour les histogrammes.
    """
    palette = ["#468fc0", "#ff8e2a", "#2ca02c", "#d62728", "#9467bd"]
    if n <= len(palette):
        return palette[:n]
    return [palette[i % len(palette)] for i in range(n)]


def _label_list_from_ids(class_ids, classe_labels):
    """
    Construit des labels au format "id - nom" pour une liste d'identifiants.
    """
    labels = []
    for c in class_ids:
        name = classe_labels.get(int(c), "Inconnu")
        labels.append(f"{int(c)} - {name}")
    return labels

# ============================================================
# ANALYSE DES ÉCHANTILLONS (POLYGONES / PIXELS)
# ============================================================


def plot_polygon_histogram(gdf, classe_col, classe_labels, out_file):
    """
    Diagramme en bâtons : nombre de polygones par classe.
    """
    counts = gdf[classe_col].value_counts().sort_index()
    classes = counts.index.tolist()
    nb_poly = counts.values

    labels = _label_list_from_ids(classes, classe_labels)

    plt.figure(figsize=(6, 5))
    colors = _default_colors(len(classes))
    bars = plt.bar(classes, nb_poly, color=colors)

    max_val = float(np.max(nb_poly)) if len(nb_poly) else 0.0
    for bar, nb in zip(bars, nb_poly):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.01,
            str(int(nb)),
            ha="center",
            va="bottom",
        )

    plt.xticks(classes, labels)
    plt.xlabel("Classe")
    plt.ylabel("Nombre de polygones")
    plt.title("Nombre de polygones par classe")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.show()


def rasterize_vector_in_memory(
    vector_path,
    ref_raster_path,
    field_name,
    dtype=gdal.GDT_Int16,
):
    """
    Rasterise une couche vecteur sur la grille d'un raster de référence.

    Paramètres
    ----------
    vector_path : str
        Chemin vers le shapefile.
    ref_raster_path : str
        Chemin vers le raster de référence.
    field_name : str
        Champ attributaire à brûler dans le raster.
    dtype : int (GDAL)
        Type GDAL de la sortie en mémoire.

    Retour
    ------
    np.ndarray
        Tableau (y, x) avec 0 en fond.
    """
    ref_ds = rw.open_image(ref_raster_path)
    if ref_ds is None:
        raise FileNotFoundError(ref_raster_path)

    nb_row, nb_col, _ = rw.get_image_dimension(ref_ds)
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()

    mem_driver = gdal.GetDriverByName("MEM")
    out_ds = mem_driver.Create("", nb_col, nb_row, 1, dtype)
    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)

    band = out_ds.GetRasterBand(1)
    band.Fill(0)
    band.SetNoDataValue(0)

    vect_ds = ogr.Open(vector_path)
    if vect_ds is None:
        raise FileNotFoundError(vector_path)

    layer = vect_ds.GetLayer()

    gdal.RasterizeLayer(
        out_ds,
        [1],
        layer,
        options=[f"ATTRIBUTE={field_name}"],
    )

    array = band.ReadAsArray()

    vect_ds = None
    ref_ds = None
    out_ds = None

    return array


def count_pixels_by_class(roi_array):
    """
    Compte le nombre de pixels par classe dans un ROI.
    """
    roi_flat = roi_array.ravel()
    roi_flat = roi_flat[roi_flat > 0]

    if roi_flat.size == 0:
        return {}

    classes, counts = np.unique(roi_flat, return_counts=True)
    return {int(c): int(n) for c, n in zip(classes, counts)}


def plot_pixel_histogram(counts_pixel, classe_labels, out_file):
    """
    Diagramme en bâtons : nombre de pixels par classe.
    """
    classes = sorted(counts_pixel.keys())
    nb_pix = [counts_pixel[c] for c in classes]
    labels = _label_list_from_ids(classes, classe_labels)

    plt.figure(figsize=(6, 5))
    colors = _default_colors(len(classes))
    bars = plt.bar(classes, nb_pix, color=colors)

    for bar, nb in zip(bars, nb_pix):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(nb)),
            ha="center",
            va="bottom",
        )

    plt.xticks(classes, labels)
    plt.xlabel("Classe")
    plt.ylabel("Nombre de pixels")
    plt.title("Nombre de pixels par classe")
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================
# ARI (SÉRIE TEMPORELLE)
# ============================================================

def calculate_ari(b03, b05):
    """
    Calcule l'indice ARI (NARI) :

        ARI = (1/B03 - 1/B05) / (1/B03 + 1/B05)
.
    """
    b03 = b03.astype(np.float32)
    b05 = b05.astype(np.float32)

    mask = (b03 == 0) | (b05 == 0)

    inv_b03 = np.divide(1.0, b03, out=np.zeros_like(b03), where=~mask)
    inv_b05 = np.divide(1.0, b05, out=np.zeros_like(b05), where=~mask)

    num = inv_b03 - inv_b05
    den = inv_b03 + inv_b05

    ari = np.divide(
        num,
        den,
        out=np.full_like(b03, np.nan),
        where=den != 0,
    )
    ari[mask] = np.nan

    return ari.astype(np.float32)


def compute_ari_stack(b03_file, b05_file):
    """
    Calcule un stack ARI (n_dates, y, x) à partir de deux rasters multibandes.
    """
    arr_b03 = rw.load_img_as_array(b03_file).astype(np.float32)
    arr_b05 = rw.load_img_as_array(b05_file).astype(np.float32)

    if arr_b03.shape != arr_b05.shape:
        raise ValueError("Les dimensions de B03 et B05 ne correspondent pas.")

    n_dates = arr_b03.shape[2]
    ari_list = []

    for i in range(n_dates):
        ari_i = calculate_ari(arr_b03[:, :, i], arr_b05[:, :, i])
        ari_list.append(ari_i)

    return np.stack(ari_list, axis=0)


def write_ari_timeseries_tif(
    ari_stack,
    ref_raster,
    out_tif,
    dates=None,
    nodata=-9999.0,
):
    """
    Écrit la série temporelle ARI (dates, y, x) en GeoTIFF float32.
    """
    ref_ds = rw.open_image(ref_raster)
    if ref_ds is None:
        raise FileNotFoundError(ref_raster)

    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()

    arr_3d = np.transpose(ari_stack, (1, 2, 0)).copy()
    arr_3d[np.isnan(arr_3d)] = nodata

    rw.write_image(
        out_filename=out_tif,
        array=arr_3d,
        transform=gt,
        projection=proj,
        driver_name="GTiff",
        gdal_dtype=gdal.GDT_Float32,
    )

    set_band_nodata(out_tif, float(nodata), band_index=1)

    if dates is not None:
        set_band_descriptions(out_tif, dates)

    ref_ds = None


def stats_by_strate(ari_stack, roi_array, strate_ids):
    """
    Statistiques ARI par strate : moyenne et écart-type pour chaque date.

    Retour
    ------
    mean_dict, std_dict : dict[int -> np.ndarray]
        Dictionnaires: id_strate -> série (n_dates,)
    """
    mean_dict = {}
    std_dict = {}

    for s in strate_ids:
        mask = roi_array == s
        if not np.any(mask):
            continue

        pixels = ari_stack[:, mask]
        mean_dict[s] = np.nanmean(pixels, axis=1)
        std_dict[s] = np.nanstd(pixels, axis=1)

    return mean_dict, std_dict


def plot_ari_timeseries(mean_dict, std_dict, strate_labels, dates, strate_ids,
                        out_file):
    """
    Trace la série temporelle moyenne d'ARI par strate avec ±1 écart-type.
    """
    dates_plot = dates[::-1]
    x = np.arange(len(dates_plot))

    plt.figure(figsize=(9, 5))

    for s in strate_ids:
        if s not in mean_dict:
            continue

        y_mean = mean_dict[s][::-1]
        y_std = std_dict[s][::-1]

        label = f"{s} - {strate_labels.get(s, 'Inconnu')}"
        plt.plot(x, y_mean, label=label)
        plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.25)

    plt.xticks(x, dates_plot, rotation=45)
    plt.xlabel("Date")
    plt.ylabel("ARI")
    plt.title("Série temporelle moyenne de l'ARI par strate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================
# CLASSIFICATION SUPERVISÉE (RANDOM FOREST)
# ============================================================

def build_predictors_stack(data_dir, dates, bands, ari_stack):
    """
    Construit le cube des prédicteurs (y, x, n_features) et les noms de variables.
    """
    band_arrays = {}

    for b in bands:
        path = os.path.join(data_dir, f"bretagne_23-24_{b}.tif")
        band_arrays[b] = rw.load_img_as_array(path).astype(np.float32)

    y_size, x_size, n_dates = band_arrays[bands[0]].shape
    if n_dates != len(dates):
        raise ValueError("Nombre de bandes (dates) != liste des dates.")

    ari_3d = np.transpose(ari_stack, (1, 2, 0)).astype(np.float32)
    ari_3d[np.isnan(ari_3d)] = -9999.0

    predictors_list = []
    feature_names = []

    for i_date in range(n_dates):
        for b in bands:
            predictors_list.append(band_arrays[b][:, :, i_date])
            feature_names.append(f"{b}_{dates[i_date]}")

    for i_date in range(n_dates):
        predictors_list.append(ari_3d[:, :, i_date])
        feature_names.append(f"ARI_{dates[i_date]}")

    predictors = np.stack(predictors_list, axis=2).astype(np.float32)

    if predictors.shape[:2] != (y_size, x_size):
        raise ValueError("Dimensions inattendues pour le cube de prédicteurs.")

    return predictors, feature_names


def write_predictors_tmp(predictors, ref_raster, out_tif):
    """
    Écrit le cube de prédicteurs (y, x, n_features) dans un GeoTIFF temporaire.
    """
    ref_ds = rw.open_image(ref_raster)
    if ref_ds is None:
        raise FileNotFoundError(ref_raster)

    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()

    rw.write_image(
        out_filename=out_tif,
        array=predictors,
        transform=gt,
        projection=proj,
        driver_name="GTiff",
        gdal_dtype=gdal.GDT_Float32,
    )

    ref_ds = None


def write_roi_tmp(roi_array, ref_raster, out_tif, nodata=0):
    """
    Écrit un ROI dans un GeoTIFF temporaire.
    """
    ref_ds = rw.open_image(ref_raster)
    if ref_ds is None:
        raise FileNotFoundError(ref_raster)

    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()

    rw.write_image(
        out_filename=out_tif,
        array=roi_array.astype(np.int16),
        transform=gt,
        projection=proj,
        driver_name="GTiff",
        gdal_dtype=gdal.GDT_Int16,
    )

    set_band_nodata(out_tif, float(nodata), band_index=1)
    ref_ds = None


def extract_samples_from_roi(predictors_tif, roi_tif, keep_labels=None):
    """
    Paramètres
    ----------
    predictors_tif : str
        GeoTIFF contenant les variables (multibandes).
    roi_tif : str
        GeoTIFF du ROI (labels).
    keep_labels : list[int] optionnel
        Si fourni, ne conserve que ces labels (ex: [2, 3, 4]).

    Retour
    ------
    X : (n, n_features)
    y : (n,)
    """
    X, Y, _t = get_samples_from_roi(raster_name=predictors_tif, roi_name=roi_tif)

    y = Y.flatten().astype(int)
    mask = y > 0

    if keep_labels is not None:
        keep = np.isin(y, np.array(keep_labels, dtype=int))
        mask = mask & keep

    return X[mask], y[mask]


def train_rf_gridsearch(
    X_train,
    y_train,
    random_state=42,
    n_splits=5,
    scoring="f1_macro",
):
    """
    Apprend un RandomForest optimisé via GridSearchCV (CV stratifiée).

    Retour
    ------
    grid : sklearn.model_selection.GridSearchCV
        L'objet GridSearch entraîné (.best_estimator_ disponible).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    param_grid = {
        "n_estimators": [50, 100, 150, 200, 300],
        "max_depth": [None, 10, 15, 20],
        "max_features": [None, "sqrt", "log2"],
        "min_samples_leaf": [1, 5],
    }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
    )

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    return grid


def evaluate_model(
    model,
    X_test,
    y_test,
    classe_labels,
    labels_eval,
    fig_dir,
    normalize_cm=True,
):
    """
    Évalue le modèle (Overall Accuracy) et produit une matrice de confusion.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
    )

    y_pred = model.predict(X_test)
    oa = accuracy_score(y_test, y_pred)

    print("Overall Accuracy :", round(oa * 100, 2), "%")

    cm_counts = confusion_matrix(y_test, y_pred, labels=labels_eval)

    labels_txt = _label_list_from_ids(labels_eval, classe_labels)

    plots.plot_cm(
        cm=cm_counts,
        labels=labels_txt,
        out_filename=None,
        normalize=normalize_cm,
        cmap="Greens",
    )


def plot_feature_importance_percent(model, feature_names, title, out_file=None):
    """
    Trace l'importance des variables en pourcentage.
    """
    importances = np.asarray(model.feature_importances_, dtype=float)
    pct = 100.0 * importances / float(importances.sum())

    order = np.argsort(pct)[::-1]

    plt.figure(figsize=(18, 10))
    plt.bar(range(len(pct)), pct[order])
    plt.xticks(
        range(len(pct)),
        [feature_names[i] for i in order],
        rotation=90,
    )
    plt.ylabel("Importance (%)")
    plt.title(title)
    plt.tight_layout()

    if out_file is not None:
        plt.savefig(out_file, dpi=200, bbox_inches="tight")

    plt.show()


def plot_importance_by_band_and_date_percent(
    model,
    feature_names,
    dates,
    bands,
    out_file_band=None,
    out_file_date=None,
):
    """
    Agrège les importances (MDI) et trace en % :
    - par bande (10 bandes Sentinel-2 + ARI)
    - par date (5 dates)
    """
    def parse_feature_name(fname):
        band, date = fname.split("_", 1)
        return band, date

    importances = np.asarray(model.feature_importances_, dtype=float)
    total = float(importances.sum())

    bands_all = list(bands) + ["ARI"]
    imp_by_band = {b: 0.0 for b in bands_all}
    imp_by_date = {d: 0.0 for d in dates}

    for imp, fname in zip(importances, feature_names):
        band, date = parse_feature_name(fname)

        if band in imp_by_band:
            imp_by_band[band] += float(imp)
        if date in imp_by_date:
            imp_by_date[date] += float(imp)

    pct_by_band = {b: 100.0 * imp_by_band[b] / total for b in bands_all}
    pct_by_date = {d: 100.0 * imp_by_date[d] / total for d in dates}

    # --- par bande
    x_band = bands_all
    y_band = [pct_by_band[b] for b in x_band]

    plt.figure(figsize=(10, 5))
    plt.bar(x_band, y_band)
    plt.xlabel("Bande (Sentinel-2 + ARI)")
    plt.ylabel("Importance (%)")
    plt.title("Importance des variables agrégée par bande")
    plt.tight_layout()

    if out_file_band is not None:
        plt.savefig(out_file_band, dpi=200, bbox_inches="tight")

    plt.show()

    # --- par date
    x_date = dates
    y_date = [pct_by_date[d] for d in x_date]

    plt.figure(figsize=(10, 5))
    plt.bar(x_date, y_date)
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Importance (%)")
    plt.title("Importance des variables agrégée par date")
    plt.tight_layout()

    if out_file_date is not None:
        plt.savefig(out_file_date, dpi=200, bbox_inches="tight")

    plt.show()


def predict_full_map(model, predictors, valid_mask, batch_size=250000):
    """
    Prédit une carte de classes à partir d'un cube de prédicteurs.

    - Les pixels invalides (valid_mask == False) sont fixés à 0.
    - La prédiction est faite en batch sur les pixels valides.
    """
    y_size, x_size, n_feat = predictors.shape

    flat = predictors.reshape(-1, n_feat)
    pred_flat = np.zeros(flat.shape[0], dtype=np.uint8)

    valid_flat = valid_mask.ravel()
    idx = np.where(valid_flat)[0]

    for start in range(0, idx.size, batch_size):
        part = idx[start: start + batch_size]
        pred_flat[part] = model.predict(flat[part]).astype(np.uint8)

    pred_map = pred_flat.reshape(y_size, x_size)
    pred_map[~valid_mask] = 0

    return pred_map.astype(np.uint8)


def write_classified_map(out_tif, class_map, ref_raster, nodata=0):
    """
    Écrit une carte classifiée.
    """
    ref_ds = rw.open_image(ref_raster)
    if ref_ds is None:
        raise FileNotFoundError(ref_raster)

    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()

    rw.write_image(
        out_filename=out_tif,
        array=class_map.astype(np.uint8),
        transform=gt,
        projection=proj,
        driver_name="GTiff",
        gdal_dtype=gdal.GDT_Byte,
    )

    set_band_nodata(out_tif, float(nodata), band_index=1)
    ref_ds = None
