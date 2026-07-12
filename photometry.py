from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.coordinates import Angle
from frb.surveys.panstarrs import Pan_STARRS_Survey
from photutils.aperture import SkyEllipticalAperture
from photutils.aperture import aperture_photometry
from photutils.aperture import SkyEllipticalAnnulus 
import numpy as np
from astropy.io import fits

#SINGLE BAND PHOTOMETRY
def measure_photometry(
    ra,
    dec,
    imsize,
    aper_a,
    aper_b,
    ann_a_in,
    ann_a_out,
    theta,
    filt,
    save_image=False):
    
    '''
    Perform elliptical aperture photometry on a Pan-STARRS image in a specified filter
       
    Parameters
       -----------
    ra: float
        right ascension in degrees
    dec: float
        declination in degrees

    imsize: float
        size of image cutout in arcsec
        
    aper_a : float
        semimajor axis of aperture in arcsec
    aper_b: float
        semiminor axis of aperture in arcsec
       
    ann_a_in: float
        semimajor axis of annulus (inner) in arcsec
    ann_a_out: float
        semimajor axis of annulus (outer) in arcsec
        
    theta: float
        angle of ellipse aperture and annulus in deg from DS9

    filt: string
        filter of the image (g, r, i, z, y)

    save_image: bool
        if True, saves Pan-STARRS image as fits file
    '''
    
    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")

    #search Pan-STARRS for galaxy within a 10'' radius
    galaxy = Pan_STARRS_Survey(coord=coord, radius=10*u.arcsec)

    #downlaod image
    image = galaxy.get_image(imsize=imsize*u.arcsec, filt=filt, timeout=120)
    
    if save_image:
        fits.HDUList([image]).writeto(f'galaxy_{filt}.fits', overwrite=True) 

    #world coordinates system, maps every x,y pixel coordinate in the image to a precise physical location on the sky
    wcs = WCS(image.header)

    # exposure time 
    exptime = image.header.get('EXPTIME')

    #zero point
    zp = image.header.get('HIERARCH FPA.ZP')



#aperture
#---------------------------------------------
    #aperture semimajor and semiminor axis
    a=aper_a*u.arcsec #semimajor
    b=aper_b*u.arcsec #semiminor

    #angle aperture is at
    theta = Angle(theta+90, 'deg') 

    #create aperture
    aper = SkyEllipticalAperture(
        coord, 
        a=a, 
        b=b, 
        theta=theta) #aperture specification

    #aperture from sky coords to pixel coords
    pix_aper = aper.to_pixel(wcs) 

    #photometry table for aperture
    source_table = aperture_photometry(image.data, pix_aper)


    #sum of source 
    source_sum = source_table['aperture_sum'][0]

    #area of source
    aperture_area = pix_aper.area

#annulus
#--------------------------------------------
    #semi major and semiminor axises for annulus
    a_out = ann_a_out
    b_out = (b/a) *a_out
    a_in = ann_a_in
    b_in = (b/a) * a_in

    #defining annulus
    annulus = SkyEllipticalAnnulus(coord, 
                               a_in=a_in*u.arcsec,
                               a_out=a_out*u.arcsec,
                               b_out=b_out*u.arcsec,
                               b_in=b_in*u.arcsec,
                               theta=theta)
    #maps annulus from sky coords to pixel coords
    pix_annulus = annulus.to_pixel(wcs)

    #photometry table for annulus
    bkg_table = aperture_photometry(image.data, pix_annulus)


    #area of annulus
    annulus_area = pix_annulus.area

    #sum of bkg
    background_sum = bkg_table['aperture_sum'][0]

    #annulus mean
    mean_annulus = background_sum/annulus_area

    #compute raw Flux
    raw_flux = source_sum -(aperture_area*mean_annulus)

    #raw magnitude
    mag_inst = -2.5*(np.log10(raw_flux/exptime)) #divided raw_flux by exptime 

    #calibrate magnitude with ZP
    mag_calibrated = zp+mag_inst

    return mag_calibrated

#MULTIBAND PHOTOMETRY     
def measure_multiband_photometry(
    ra,
    dec,
    imsize,
    aper_a,
    aper_b,
    ann_a_in,
    ann_a_out,
    theta,
    save_image=False
):
    '''
    Perform elliptical aperture photometry on Pan-STARRS images in multiple filters

    Parameters
    ----------
    ra: float
        right ascension in degrees

    dec: float
        declination in degrees

    imsize: float
        size of image cutout in arcsec

    aper_a: float
        semimajor axis of aperture in arcsec

    aper_b: float
        semiminor axis of aperture in arcsec

    ann_a_in: float
        inner semimajor axis of annulus in arcsec

    ann_a_out: float
        outer semimajor axis of annulus in arcsec

    theta: float
        angle of ellipse from DS9 in degrees

    save_image: bool
        if True, saves Pan-STARRS images as fits file

    Returns
    -------
    magnitudes: dict
        calibrated magnitudes for each Pan-STARRS filter
    '''

    #usable filters
    filters = ['g', 'r', 'i', 'z', 'y']

    #dictionary of filters and their magnitudes
    magnitudes = {}

    for filt in filters:

        mag = measure_photometry(
            ra=ra,
            dec=dec,
            imsize=imsize,
            aper_a=aper_a,
            aper_b=aper_b,
            ann_a_in=ann_a_in,
            ann_a_out=ann_a_out,
            theta=theta,
            filt=filt,
            save_image=save_image
        )

        magnitudes[filt] = mag

    return magnitudes   
