import numpy as np
from numpy.fft import fft2
from numpy.fft import ifft2
import numpy.random
from numpy.random import standard_normal
import scipy.constants as sc
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar


# SEED

seed = 17
numpy.random.seed(seed)


# QUANTITIES

h = sc.h
m_e = sc.m_e
e = sc.e
c = sc.c
pi = np.pi


def calc_Ronchigram(imdim, simdim, C10_mag, C12_mag, C21_mag, C23_mag, C10_ang, C12_ang, C21_ang, C23_ang, I, b, t):
    """Takes the aberration coefficient magnitudes mentioned above (in Krivanek notation) and returns a NumPy array of
    the resulting Ronchigram.

    parameters
    imdim: output Ronchigrams have dimensions imdim x imdim px, int
    simdim: maximum alpha (tilt) angle in Ronchigram/rad, num (i.e. maximum angle of inclination of electrons to normal)
    Cnm_mag: magnitude of aberration of order n and rotational symmetry m (in Krivanek notation)/m, num
    # TODO: make sure C10's magnitude is actually in m
    Cnm_ang: angle of aberration of order n and rotational symmetry m (in Krivanek notation)/rad, num
    # PEAK: parameter dictating the amount of Poisson noise applied to the Ronchigram, num
    I: quoted electron current in Ronchigram generation/A, num
    b: fraction of I that reaches the detector, num
    t: Ronchigram aquisition time/s, num

    returns
    Ronchigram, numpy.ndarray
    """

    # n is order of geometrical aberration, m is rotational symmetry
    n_list = (1, 1, 2, 2)
    m_list = (0, 2, 1, 3)

    # These are the magnitudes of each aberration Cn,m (Cn,m are aberration names in Krivanek notation (Krivanek, Dellby
    # and Lupini, 1999)). The terms below are named in Table 1 of (Kirkland, 2011), which is compiled from (Krivanek,
    # Dellby and Lupini, 1999) and (Krivanek et al., 2008).
    mag_list = (C10_mag,    # C1,0 magnitude/m (defocus)
                C12_mag,    # C1,2 magnitude/m (2-fold astigmatism)
                C21_mag,    # C2,1 magnitude/m (axial coma)
                C23_mag)    # C2,3 magnitude/m (3-fold astigmatism)

    # TODO: discern what type of lens (e.g. over or under-focussed) the signs allude to in the later aberration function
    #   equation
    # TODO: check if chromatic aberration should be accounted for

    # Orientation angles phi_n,m for each aberration; follows same convention as np.arctan2
    ang_list = (C10_ang,    # C1,0 angle/rad
                C12_ang,    # C1,2 angle/rad
                C21_ang,    # C2,1 angle/rad
                C23_ang)    # C2,3 angle/rad

    def chi_term(al, phi, av, n, m, mag, ang):
        """Takes an aberration and determines its contribution to chi (aberration function) for a given convergence angle 
        and azimuthal angle.

        al: convergence angle/rad, num
        phi: azimuthal angle/rad, num
        av: accelerating voltage/V (not relativistically corrected), num
        n: order of aberration, int
        m: degree of aberration, int
        mag: magnitude of aberration/unit, num
        ang: angle of orientation of aberration/rad, num

        returns: numpy.float64
        """

        wavlen = calc_wavlen(av)

        # Term of aberration function equation mentioned in (Schnitzer, Sung and Hovden, 2019), originally from
        #   Krivanek et al., 1999)
        return 2*np.pi/wavlen * mag * al**(n+1) * np.cos(m*(phi-ang)) / (n+1)


    def chi_value(al, phi, av, n_list, m_list, mag_list, ang_list):
        """For a given convergence angle and azimuthal angle, takes aberrations information stored in lists (index
        corresponds to specific aberration) and calculated aberration function value.

        al: convergence angle/rad, num
        phi: azimuthal angle/rad, num
        av: accelerating voltage/V (not relativistically corrected), num
        n: orders of aberration, list
        m: degree of aberration, list
        mag: magnitude of aberration/unit, list
        ang: angle of orientation of aberration/rad, list

        returns: numpy.float64
        """

        chi_terms = [chi_term(al, phi, av, n, m, mag, ang) for n, m, mag, ang in zip(n_list, m_list, mag_list, ang_list)]

        # Aberration function equation mentioned in (Schnitzer, Sung and Hovden, 2019)
        return sum(chi_terms)


    def chi_grid(imdim, al_max, av, n_list, m_list, mag_list, ang_list):
        """Creates an imdim by imdim array of aberration function values for convergence angles (al) with components of magnitudes
        between -al_max and +al_max, for a given set of aberrations; returns an equally sized array containing the
        convergence angle radii at each point, too.

        imdim: length of side of image in pixels (or of array in elements), num
        al_max: magnitude of the maximum magnitude of a convergence angle component/rad, num
        av: accelerating voltage/V (not relativistically corrected), num
        n: orders of aberration, list
        m: degree of aberration, list
        mag: magnitude of aberration/unit, list
        ang: angle of orientation of aberration/rad, list

        returns: tuple(chi_array, al_array)
        chi_array: array of aberration function values, numpy.ndarray
        al_rr: array of converge angle radii/rad, numpy.ndarray
        """
        # The below was written with the assistance of (Schnitzer, 2020c)

        # 1D vector containing convergence angles' possible x- and y- component values. The third argument is
        # imdim because the images in question contain imdim by imdim pixels.
        al_components = np.linspace(-al_max, al_max, imdim)

        # 2d grid of convergence angle vectors formed of different combinations of x- and y- components above.
        al_grid = np.meshgrid(al_components, al_components)

        al_xx, al_yy = al_grid

        al_rr = np.sqrt(al_xx ** 2 + al_yy ** 2)

        # Finding azimuthal angle at each grid point, using arctan2 to account for quadrant position.
        phi = np.arctan2(al_yy, al_xx)

        chi_array = chi_value(al_rr, phi, av, n_list, m_list, mag_list, ang_list)

        return (chi_array, al_rr)


    # CALCULATING THE INTERACTION PARAMETER SIGMA MENTIONED IN (Schnitzer, Sung and Hovden, 2019)

    def calc_wavlen(av):
        """Calculates relativistically corrected electron wavelength given accelerating voltage (Williams and Carter, 2009).
        N.B. requires h/Js, m_e (electron rest mass/kg), e (electron charge/J) and c/ms**-1 to be assigned globally as
        variables.

        av: accelerating voltage/V (not relativistically corrected), num

        returns: wavelength/m, num
        """

        # This has been shown on paper (by me) to be approximately equal to the lambda equation in shifted_ronchigram_m.txt,
        # which I am unsure uses the relativistically corrected electron wavelength.
        wavlen = h / np.sqrt(2 * m_e * e * av * (1 + e * av / (2 * m_e * c ** 2)))

        # print(f"Wavelength is {wavlen*10**12}pm")
        return wavlen


    # SIMULATION PARAMETERS FROM (Schnitzer, 2020b) and (Schnitzer, 2020c)

    imdim = imdim    # Output Ronchigram size/pixels

    simdim = simdim    # Simulation radius in reciprocal space/rad (essentially, the maximum tilt angle)
    # 50-80mrad is a lot more realistic today, according to Chen


    # NOISE CALCULATIONS FROM (Schnitzer, 2020c)

    nnoise = 1  # Presumably the number of noise functions
    noisefact = 16

    assert imdim % noisefact == 0
    noise_kernel_size = int(imdim/noisefact)    # 256 ->32, 512 ~ 32, 1024 -> 128   # Comment meaning

    assert imdim % noise_kernel_size == 0
    resize_factor = int(imdim/noise_kernel_size)

    noise_fn = np.zeros((noise_kernel_size, noise_kernel_size))
    for i in range(nnoise):
        noise_fn += standard_normal(size=(noise_kernel_size, noise_kernel_size))

    # todo: figure out where they cut off frequencies above Kmax/2

    noise_fn /= nnoise
    noise_fn += 1
    noise_fn /= 2

    new_shape = tuple([resize_factor * i for i in noise_fn.shape])
    noise_fun = np.array(Image.fromarray(noise_fn).resize(size=new_shape))


    # CALCULATING THE TRANSMISSION FUNCTION PSI_T(X)

    av = 300 * 10**3    # Acceleration voltage/V (not relativistically corrected)
    kev = av / 1000 # Electron energy/keV (not relativistically corrected)

    # Chi is the aberration function
    chi_array = chi_grid(imdim, simdim, av, n_list, m_list, mag_list, ang_list)[0]
    al_rr = chi_grid(imdim, simdim, av, n_list, m_list, mag_list, ang_list)[1]

    fft_psi_p = fft2(np.exp(-1j*chi_array))    # (Schnitzer, 2020a)

    # Schnitzer's, from (Schnitzer, 2020c)
    inter_param_schnitzer = 2*pi/(calc_wavlen(av)*kev/e*1000)*(m_e*c**2+kev*1000)/(2*m_e*c**2+kev*1000)
    exp_part = np.exp(-1j * np.pi/4 * inter_param_schnitzer * noise_fun / (1.7042*10**-12))   # Schnitzer's

    # Transmission wavefunction under the eikonal approximation (Schnitzer, Sung and Hovden, 2019)
    psi_t = fft_psi_p * exp_part


    # CALCULATING THE RONCHIGRAM

    inverse = ifft2(psi_t)  # (Schnitzer, Sung and Hovden, 2020), (Schnitzer, 2020c)

    ronch = abs(inverse)**2 # (Schnitzer, Sung and Hovden, 2020)


    # NORMALISING THE RONCHIGRAM (TO ARRAY VALUES BETWEEN 0 AND 1)

    # (Schnitzer, 2020c)
    ronch -= np.amin(ronch)
    ronch /= np.amax(ronch)

    # print(np.amax(ronch))

    # APPLYING OBJECTIVE APERTURE

    # For detector Poisson noise, I previously followed https://stackoverflow.com/questions/19289470/adding-poisson-noise-to-an-image 
    # for assistance, except with 255 replaced by 1 because normalisation of ronch somewhere above lead to array elements between 0 and 1. I took
    # yuxiang.li's suggestion, but only their upper one, because the lower one looked like it simply added the Poisson noise
    # to the image, which didn't seem right since Poisson noise isn't additive. 
    # Now, however, I am using my own formulation of Poisson noise derived in the 02/12/2021 entry of my lab book.

    # I is quoted electron current/A (Angus mentioned picoamps being a typical scale)
    # b is fraction of I that reaches the detector
    # t is Ronchigram collection time/s (I overheard Chen and Angus perhaps mention 1s)
    # See 02/12/2021 entry to lab book for derivation of the below
    lam = I * t / sc.e * b * ronch / np.sum(ronch)
    rng = np.random.default_rng()   # Random number generator that NumPy documentation recommends
    ronch = rng.poisson(lam)

    # print(np.amax(ronch))

    return ronch


if __name__ == "__main__":

    # RONCHIGRAM CALCULATION

    mag_list = (10 * 10**-9,    # C1,0 magnitude/m (defocus)
                50 * 10**-9,    # C1,2 magnitude/m (2-fold astigmatism)

                1000 * 10**-9,    # C2,1 magnitude/m (axial coma)
                1000 * 10**-9)    # C2,3 magnitude/m (3-fold astigmatism)

    ang_list = (0,        # C1,0 angle/rad
                0,    # C1,2 angle/rad
                0,    # C2,1 angle/rad
                0)    # C2,3 angle/rad

    imdim = 1024
    simdim = 150 * 10**-3

    ronch = calc_Ronchigram(imdim, simdim, *mag_list, *ang_list, I=10**-9, b=1, t=1)

    # DEPICTING THE RONCHIGRAM

    # todo: consider changing the interpolation in order to match pixels for a better image
    # source: https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
    # plt.imshow(ronch, cmap="gray", interpolation="nearest")

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.imshow(ronch, cmap="gray", interpolation="nearest")

    scale = 2*simdim/imdim    # radians per pixel
    scale_mrad = scale * 10**3  # mrad per pixel
    scalebar = ScaleBar(scale_mrad, units="m", dimension="si-length")

    ax.add_artist(scalebar)

    dominantcnm = "c12"
    saveFig = False

    if saveFig:
        plt.savefig(f"/media/rob/hdd2/james/simulations/exampleRonchigrams/toFindNegligibleRanges/{dominantcnm}Dominant/c10_{mag_list[0]/10**-9}_c12_{mag_list[1]/10**-9}_c21_{mag_list[2]/10**-9}_c23_{mag_list[3]/10**-9}.png")

    plt.show()


    # REFERENCES

    # Jiang, W. and Chiu, W. (2001) “Web-based Simulation for Contrast Transfer Function and Envelope Functions,”
    # Microscopy and Microanalysis, 7(4). doi:10.1007/S10005-001-0004-4.

    # Kirkland, E.J. (2011) “On the optimum probe in aberration corrected ADF-STEM,” Ultramicroscopy, 111(11).
    # doi:10.1016/j.ultramic.2011.09.002.

    # Krivanek, O.L. et al. (2008) “An electron microscope for the aberration-corrected era,” Ultramicroscopy, 108(3).
    # doi:10.1016/j.ultramic.2007.07.010.

    # Krivanek, O.L., Dellby, N. and Lupini, A.R. (1999) “Towards sub-Å electron beams,” Ultramicroscopy, 78(1–4), pp. 1–11.

    # Schnitzer, N. (2020a) ronchigram-matlab/calculate_probe.m at master · noahschnitzer/ronchigram-matlab. Available at:
    # https://github.com/noahschnitzer/ronchigram-matlab/blob/master/simulation/calculate_probe.m (Accessed: October 21,
    # 2021).

    # Schnitzer, N. (2020b) ronchigram-matlab/example.m at master · noahschnitzer/ronchigram-matlab. Available at:
    # https://github.com/noahschnitzer/ronchigram-matlab/blob/master/misc/example.m (Accessed: October 21, 2021).

    # Schnitzer, N. (2020c) ronchigram-matlab/shifted_ronchigram.m at master · noahschnitzer/ronchigram-matlab. Available
    # at: https://github.com/noahschnitzer/ronchigram-matlab/blob/master/simulation/shifted_ronchigram.m (Accessed: October
    # 21, 2021).

    # Schnitzer, N., Sung, S.H. and Hovden, R. (2019) “Introduction to the Ronchigram and its Calculation with
    # Ronchigram.com,” Microscopy Today, 27(3), pp. 12–15. doi:10.1017/S1551929519000427.

    # Schnitzer, N., Sung, S.H. and Hovden, R. (2020) “Optimal STEM Convergence Angle Selection Using a Convolutional
    # Neural Network and the Strehl Ratio,” Microscopy and Microanalysis, 26(5). doi:10.1017/S1431927620001841.

    # Williams, D.B. and Carter, C.B. (2009) Transmission electron microscopy: a textbook for materials science. 2nd
    # edn. New York: Springer.