from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.background import MedianBackground, Background2D
from astropy.convolution import convolve
from photutils.segmentation import (
    make_2dgaussian_kernel,
    detect_sources
)
import numpy as np
from photutils.segmentation import SourceCatalog
from astropy import units as u
import matplotlib.pyplot as plt


def image_segmentation(
    fits_file,
    threshold_sigma=3,
    n_pix=300,
    box_size=(100,100),
    filter_size=(3,3),
    plot=True
):
    """
    Performs source detection on an astronomical FITS image.

    Parameters
    ----------
    fits_file : str
        Path to any FITS image.

    threshold_sigma : float
        Detection threshold in units of background RMS.

    n_pix : int
        Minimum connected pixels for detection.

    box_size : tuple
        Background2D mesh size.

    filter_size : tuple
        Background smoothing size.

    plot : bool
        Display segmentation map.

    Returns
    -------
    segment_map : photutils.segmentation.SegmentationImage
        Segmentation map of detected sources.
    """

    #load FITS image
    image = fits.getdata(fits_file) #all 
    image = np.nan_to_num(image) #all Nan values are assigned 0. infinity points are given the highest value pixel
    print(image.shape)

    #estimate background
    bkg_estimator = MedianBackground()

    bkg = Background2D(
        image,
        box_size,
        filter_size=filter_size,
        bkg_estimator=bkg_estimator
    )

    #subtract background from image
    image_sub = image - bkg.background


    #noise is estimated by finding the median and std of the noise pixel values
    _, median, std = sigma_clipped_stats(image_sub)


    #create smoothing kernel
    kernel = make_2dgaussian_kernel(
        3.0,
        size=5
    )


    #smooth image
    convolved_data = convolve(
        image_sub,
        kernel
    )


    # detection threshold 
    threshold = median + threshold_sigma * std


    # Detect sources
    segment_map = detect_sources(
        convolved_data,
        threshold,
        npixels=n_pix
    )

    #utilzie source catalog to find properties of the galaxy on image 
    source_cat = SourceCatalog(image_sub, segment_map, convolved_data=convolved_data)
    

    
  #visualize detected sources
    if plot:

        fig, ax = plt.subplots(figsize=(8,8))

        # show the original image
        ax.imshow(image_sub, origin='lower', cmap='gray')

    #overlay segmentation regions
        if segment_map is not None:
            segment_map.imshow(
                ax=ax,
                alpha=0.5
            )

        #label each segment at its centroid
        for source in source_cat:

            x = source.x_centroid
            y = source.y_centroid

            ax.text(
                x,
                y,
                str(source.label),
                color='white',
                fontsize=10
            )

        ax.set_title("Detected Sources")
        plt.show()

    return segment_map, source_cat, image_sub
