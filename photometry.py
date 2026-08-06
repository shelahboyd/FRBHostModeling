from photutils.aperture import aperture_photometry
from photutils.aperture import EllipticalAnnulus 
import numpy as np



def measure_photometry(image,
                       catalog, 
                       source_id,
                       exptime,
                       zp):
    
    

    target_source = catalog[source_id]

    source_aper = target_source.kron_aperture

    source_positions = source_aper.positions

    bkg_positions = source_positions

    bkg_aper = EllipticalAnnulus(
            bkg_positions,
            a_in = source_aper.a*3,
            a_out = source_aper.a*4,
            b_in = source_aper.b*3,
            b_out = source_aper.b*4,
            theta = source_aper.theta)

    source_table = aperture_photometry(image, source_aper)

    source_sum = source_table['aperture_sum'][0]

    source_area = source_aper.area

    bkg_table = aperture_photometry(image, bkg_aper)

    bkg_sum = bkg_table['aperture_sum'][0]

    bkg_area = bkg_aper.area

    bkg_mean = bkg_sum/bkg_area

    raw_flux = source_sum - (bkg_mean*source_area)


    mag_inst = -2.5*np.log10(raw_flux/exptime)
             
    if raw_flux <= 0:
        print(f'Raw flux, {raw_flux}, is negative or 0')
        
        return raw_flux, np.nan
             
    mag_calibrated = zp+ mag_inst

    return raw_flux, mag_calibrated
