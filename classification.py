# -*- coding: utf-8 -*-
"""
@author: marc lang
"""

from osgeo import gdal, ogr
import numpy as np
import read_and_write as rw
import os


def rasterization(in_vector, ref_image, out_image, field_name, dtype=None):
    """
    Rasterize a vector layer on a reference raster using an attribute field.

    Parameters
    ----------
    in_vector : str
        Path to vector file (shapefile / geopackage).
    ref_image : str
        Path to reference raster.
    out_image : str
        Output raster path.
    field_name : str
        Attribute field to burn.
    dtype : str or None
        Optional type name: "uint16", "int16", "float32", etc.
    """

    # Mapping dtype string -> GDAL type
    dtype_map = {
        "uint16": gdal.GDT_UInt16,
        "int16": gdal.GDT_Int16,
        "uint32": gdal.GDT_UInt32,
        "int32": gdal.GDT_Int32,
        "float32": gdal.GDT_Float32,
        "float64": gdal.GDT_Float64,
    }

    gdal_dtype = dtype_map.get(dtype, gdal.GDT_UInt16)

    # Open reference raster
    ref_ds = gdal.Open(ref_image)
    if ref_ds is None:
        raise RuntimeError(f"Impossible d'ouvrir l'image de référence : {ref_image}")

    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    xsize = ref_ds.RasterXSize
    ysize = ref_ds.RasterYSize

    # Create output raster
    os.makedirs(os.path.dirname(out_image), exist_ok=True)

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        out_image,
        xsize,
        ysize,
        1,
        gdal_dtype,
        options=["COMPRESS=LZW"]
    )

    out_ds.SetGeoTransform(gt)
    out_ds.SetProjection(proj)

    band = out_ds.GetRasterBand(1)
    band.Fill(0)
    band.SetNoDataValue(0)

    # Open vector
    vec_ds = ogr.Open(in_vector)
    if vec_ds is None:
        raise RuntimeError(f"Impossible d'ouvrir le vecteur : {in_vector}")

    layer = vec_ds.GetLayer()

    # Rasterize using attribute
    gdal.RasterizeLayer(
        out_ds,
        [1],
        layer,
        options=[f"ATTRIBUTE={field_name}"]
    )

    # Close datasets
    out_ds.FlushCache()
    out_ds = None
    vec_ds = None
    ref_ds = None


def get_samples_from_roi(raster_name, roi_name, value_to_extract=None,
                         bands=None, output_fmt='full_matrix'):
    '''
    The function get the set of pixel of an image according to an roi file
    (raster). In case of raster format, both map should be of same
    size.

    Parameters
    ----------
    raster_name : string
        The name of the raster file, could be any file GDAL can open
    roi_name : string
        The path of the roi image.
    value_to_extract : float, optional, defaults to None
        If specified, the pixels extracted will be only those which are equal
        this value. By, defaults all the pixels different from zero are
        extracted.
    bands : list of integer, optional, defaults to None
        The bands of the raster_name file whose value should be extracted.
        Indexation starts at 0. By defaults, all the bands will be extracted.
    output_fmt : {`full_matrix`, `by_label` }, (optional)
        By default, the function returns a matrix with all pixels present in the
        ``roi_name`` dataset. With option `by_label`, a dictionnary
        containing as many array as labels present in the ``roi_name`` data
        set, i.e. the pixels are grouped in matrices corresponding to one label,
        the keys of the dictionnary corresponding to the labels. The coordinates
        ``t`` will also be in dictionnary format.

    Returns
    -------
    X : ndarray or dict of ndarra
        The sample matrix. A nXd matrix, where n is the number of referenced
        pixels and d is the number of variables. Each line of the matrix is a
        pixel.
    Y : ndarray
        the label of the pixel
    t : tuple or dict of tuple
        tuple of the coordinates in the original image of the pixels
        extracted. Allow to rebuild the image from `X` or `Y`
    '''

    # Get size of output array
    raster = rw.open_image(raster_name)
    nb_col, nb_row, nb_band = rw.get_image_dimension(raster)

    # Get data type
    band = raster.GetRasterBand(1)
    gdal_data_type = gdal.GetDataTypeName(band.DataType)
    numpy_data_type = rw.convert_data_type_from_gdal_to_numpy(gdal_data_type)

    # Check if is roi is raster or vector dataset
    roi = rw.open_image(roi_name)

    if (raster.RasterXSize != roi.RasterXSize) or \
            (raster.RasterYSize != roi.RasterYSize):
        print('Images should be of the same size')
        print('Raster : {}'.format(raster_name))
        print('Roi : {}'.format(roi_name))
        exit()

    if not bands:
        bands = list(range(nb_band))
    else:
        nb_band = len(bands)

    #  Initialize the output
    ROI = roi.GetRasterBand(1).ReadAsArray()
    if value_to_extract:
        t = np.where(ROI == value_to_extract)
    else:
        t = np.nonzero(ROI)  # coord of where the samples are different than 0

    Y = ROI[t].reshape((t[0].shape[0], 1)).astype('int32')

    del ROI
    roi = None  # Close the roi file

    try:
        X = np.empty((t[0].shape[0], nb_band), dtype=numpy_data_type)
    except MemoryError:
        print('Impossible to allocate memory: roi too large')
        exit()

    # Load the data
    for i in bands:
        temp = raster.GetRasterBand(i + 1).ReadAsArray()
        X[:, i] = temp[t]
        del temp
    raster = None  # Close the raster file

    # Store data in a dictionnaries if indicated
    if output_fmt == 'by_label':
        labels = np.unique(Y)
        dict_X = {}
        dict_t = {}
        for lab in labels:
            coord = np.where(Y == lab)[0]
            dict_X[lab] = X[coord]
            dict_t[lab] = (t[0][coord], t[1][coord])

        return dict_X, Y, dict_t
    else:
        return X, Y, t,
