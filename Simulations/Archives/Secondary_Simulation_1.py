import numpy as np
# from numpy.fft import fft2
from torch.fft import fft2
# from scipy.signal import convolve2d
from torch.nn.functional import conv2d
import scipy.constants as sc
from numpy.random import standard_normal
from PIL import Image
import matplotlib.pyplot as plt
import torch
# todo: put units wherever applicable

# DEVICE CONFIGURATION
GPU = 0
device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

# I am using "watch nvidia-smi" and "htop" to monitor the GPUs, but the below is some added assurance - if anyone 
# has any torch Tensor on the GPU I am trying to work on, the rest of the script won't run.
assert torch.cuda.memory_allocated(GPU) == 0
assert torch.cuda.memory_reserved(GPU) == 0

# Unrelativistically-corrected accelerating voltage/V
av = 300 * 10**3

# h in Equation 9b of Sawada et al., 2008; negligible when defocus and 2-fold astigmatism dominate in the CTF
# todo: will keep as 0 since aberrations will be randomly generated from larger ranges for these first order (re:
#   geometrical order, see Table A.1 of paper for the distinction between geometrical aberration order
#   and wave aberration order) aberrations will be drawn from much larger ranges than the second order ones. There may
#   be cases where the first-order and second-order aberrations are comparable and perhaps the former don't domintate
#   such that h is negligible, but such simulations will be few, so I will leave h as 0 for now
h = 0

# Indices re: defocus, 2-fold astigmatism, axial coma, 3-fold astigmatism. Magnitudes and angles are the same as from
# Ronchigram_Simulations/Simulation_3.py, at least as of 23/11/2021
# Note: m+n-1 = geometrical aberration order, see (Sawada et al., 2008)
m_list = [1, 2, 2, 3]
n_list = [1, 0, 1, 0]
# todo: replace cmn for defocus and 2-fold astigmatism below with much larger values, although ranges will be drawn from
#   anyway
# todo: check if defocus cmn is /rad, although this is a minor point because I am fairly sure it is
cmn_list = [9 * 10**-10, 29 * 10**-9, 73 * 10**-9, 76*10**-9]    # cmn/m
theta_cmn_list = [0, 0.995, 1.536, 1.134]    # theta_cmn/rad. defocus rotational symmetry 0, hence 0 angle


# ABERRATION PHASE FUNCTION FROM SAWADA ET AL., 2008

# todo: be very clear in the below convention in any write-ups
# Cmn here is not the same as Cmn in Krivanek notation, nor are the m and n here
# equal to the degree of symmetry and the order of the geometrical aberration
# described by Krivanek notation - see the 21/11/2021 entry of your lab book. In
# short, m+n here is the order of a wave aberration, whose corresponding
# geometrical aberration has an order of m+n-1; it seems that, for example, 2-fold
# astigmatism causes both geometrical and wave aberration, which explains why a wave
# aberration has a corresponding geometrical aberration - again, see lab book for more.
# Note, below, cmn is not Cmn, but instead (I assume) describes the magnitude of the
# aberration denoted by Cmn; theta_cmn (I assume) describes its orientation.
# Cmn = cmn * np.exp((m-n) * 1j * theta_cmn)
def calculate_Cmn(m, n, cmn, theta_cmn):
    """For a given aberration, calculates Cmn as defined in Table A.1 of the paper by Sawada et al., 2008. This
    function requires numpy to be imported as np.

    :param m: m-value of aberration, int
    :param n: n-value of aberration, int
    :param cmn: magnitude of aberration/m, num
    # todo: check if is magnitude, check if always /m (minor todo)
    :param theta_cmn: angle/rad from Table A.1 of said paper associated with the aberration, num

    :return Cmn: float
    """

    Cmn = cmn * np.exp((m-n) * 1j * theta_cmn)

    return Cmn

def calculate_wavelength(av):
    """Calculates relativistically corrected electron wavelength given accelerating voltage (Williams and Carter, 2009).
    Requires scipy.constants to be imported as sc.

    :param av: accelerating voltage/V (not relativistically corrected), num

    :return wavelength: wavelength/m, float
    """

    wavelength = sc.h / np.sqrt(2 * sc.m_e * sc.e * av * (1 + sc.e * av / (2 * sc.m_e * sc.c**2)))

    return wavelength

# general_gamma_term = 1/(m+n) * Cmn * np.conj(omega)**m * omega**n
def general_gamma_term(m, n, cmn, theta_cmn, omega):
    """Returns a gamma term in the series gamma defined by Equations 1 and A.1 of (Sawada et al., 2008). Each term is
    for a different aberration. Requires numpy to be imported as np.

    :param m: m-value of aberration, int
    :param n: n-value of aberration, int
    :param Cmn: Cmn/m of aberration, complex num
    :param omega: omega/rad, complex num
    # todo: check units of Cmn and omega (minor todo)

    :return general_gamma_term: general gamma term/m, float
    # todo: check unit of general gamma term
    """

    omega = omega.cpu()
    general_gamma_term = 1/(m+n) * calculate_Cmn(m, n, cmn, theta_cmn) * np.conj(omega)**m * omega**n

    return general_gamma_term

def chi_aberration_phase_function(av, m_list, n_list, cmn_list, theta_cmn_list, omega):
    """Returns the aberration phase function in Equation 5 of (Sawada et al.,
    2008). Note, m and n are as defined in the aforementioned paper, as are cmn
    and theta_cmn. In the lists passed as arguments, the same aberration has values
    at the same index in each list.

    :param av: accelerating voltage/V (not relativistically corrected), num
    :param m_list: list of m-values of aberrations, list[ints]
    :param n_list: list of n values of aberrations, list[ints]
    :param cmn_list: list of magnitudes/m of aberrations, list[num]
    :param theta_cmn_list: list of angles/rad associated with aberrations in Table A.1, list[num]
    :param omega: variable omega/rad mentioned by Sawada et al., 2008, complex num

    :return chi_aberration_phase_function: aberration phase function (dimensionless), float
    """

    assert len(m_list) == len(n_list) == len(cmn_list) == len(theta_cmn_list)   # (== number of aberrations)

    # Gamma, equal to the left-hand sum in the parentheses of Equation 1 of (Sawada et al., 2008). In Appendix A, this
    # is referred to using the symbol gamma.
    gamma = 0
    for i in range(len(m_list)):
        gamma += general_gamma_term(m_list[i], n_list[i], cmn_list[i], theta_cmn_list[i], omega)

    wave_aberration_function = 1 / 2 * (gamma + np.conj(gamma))
    wave_aberration_function = wave_aberration_function.real
    print(wave_aberration_function.dtype)

    wavelength = calculate_wavelength(av)

    chi_aberration_phase_function = 2 * np.pi / wavelength * wave_aberration_function

    return chi_aberration_phase_function

# Even part of the aberration phase function, equation for which is given in
# https://math.stackexchange.com/questions/5274/how-do-i-divide-a-function-into-even-and-odd-sections -
# look up what even and odd functions are.
# todo: k is "reciprocal coordinates "bold" k = (u,v)" according to (Sawada et al., 2008).
#   omega = wavelength * (u + iv), where i is the square root of -1. This is how
#   omega and k, although I think k is magnitude of bold "k" are connected. In fact, the only functions in the intensity equation
#   somewhere below are of G and k, and G is a function of omega, so I might express
#   the wave aberration function in terms of omega rather than k.
# FIXME: going to assume that the even part of chi_k (aberration phase function in
#   terms of k) equals the even part of chi_omega (aberration phase function in
#   terms of omega).
# todo: make sure that the even part of aberration function w.r.t. k equals even part of aberration function w.r.t. omega

def even_part_chi_abberation_phase_function(av, m_list, n_list, cmn_list, theta_cmn_list, omega):
    """Calculates even part of aberration phase function in Equation 5 of (Sawada et al., 2008). Equation for even
    part of function comes from
    https://math.stackexchange.com/questions/5274/how-do-i-divide-a-function-into-even-and-odd-sections.

    :param av: accelerating voltage/V (not relativistically corrected), num
    :param m_list: list of m-values of aberrations, list[ints]
    :param n_list: list of n values of aberrations, list[ints]
    :param cmn_list: list of magnitudes/m of aberrations, list[num]
    :param theta_cmn_list: list of angles/rad associated with aberrations in Table A.1, list[num]
    :param omega: variable omega/rad mentioned by Sawada et al., 2008, complex num

    :return even_part: even part of aberration phase function (dimensionless), float
    """
    # todo: make sure that the even part of aberration function w.r.t. k equals even part of aberration function w.r.t.
    #   omega

    wavelength = calculate_wavelength(av)

    even_part = (chi_aberration_phase_function(wavelength, m_list, n_list, cmn_list, theta_cmn_list, omega) +
                 chi_aberration_phase_function(wavelength, m_list, n_list, cmn_list, theta_cmn_list, -omega)) / 2

    return even_part


def odd_part_chi_abberation_phase_function(av, m_list, n_list, cmn_list, theta_cmn_list, omega):
    """Calculates odd part of aberration phase function in Equation 5 of (Sawada et al., 2008). Equation for even
    part of function comes from
    https://math.stackexchange.com/questions/5274/how-do-i-divide-a-function-into-even-and-odd-sections.

    :param av: accelerating voltage/V (not relativistically corrected), num
    :param m_list: list of m-values of aberrations, list[ints]
    :param n_list: list of n values of aberrations, list[ints]
    :param cmn_list: list of magnitudes/m of aberrations, list[num]
    :param theta_cmn_list: list of angles/rad associated with aberrations in Table A.1, list[num]
    :param omega: variable omega/rad mentioned by Sawada et al., 2008, complex num

    :return odd_part: odd part of aberration phase function (dimensionless), float
    """
    # todo: make sure that the even part of aberration function w.r.t. k equals even part of aberration function w.r.t.
    #   omega

    wavelength = calculate_wavelength(av)

    odd_part = (chi_aberration_phase_function(wavelength, m_list, n_list, cmn_list, theta_cmn_list, omega) -
                 chi_aberration_phase_function(wavelength, m_list, n_list, cmn_list, theta_cmn_list, -omega)) / 2

    return odd_part

def calculate_G(m_list, n_list, cmn_list, theta_cmn_list, omega):
    """Calculates G, complex description of geometric aberration, from Equation 3 of (Sawada et al., 2008).

    :param m_list: list of m-values of aberrations, list[ints]
    :param n_list: list of n-values of aberrations, list[ints]
    :param cmn_list: list of magnitudes/m of aberrations, list[num]
    :param theta_cmn_list: list of angles/rad associated with aberrations in Table A.1, list[num]
    :param omega: variable omega/rad mentioned by Sawada et al., 2008, complex num

    :return G: complex description of geometric aberration/m, float
    # todo: check unit of G (minor todo)
    """

    assert len(m_list) == len(n_list) == len(cmn_list) == len(theta_cmn_list)   # (== number of aberrations)

    G = 0
    for i in range(len(m_list)):
        m, n, cmn, theta_cmn = (m_list[i], n_list[i], cmn_list[i], theta_cmn[i])
        Cmn = calculate_Cmn(m, n, cmn, theta_cmn)

        G += 1 / (m+n) * (n * np.conj(Cmn) * omega**m * np.conj(omega)**(n-1) + m * Cmn * np.conj(omega)**(m-1) *
                          omega**n)

    return G

def calculate_CTF(av, m_list, n_list, cmn_list, theta_cmn_list, alpha_tilt_angle_max, imdim, h):
    """Calculates the CTF in Equation 9b of (Sawada et al., 2008). Includes Fourier transform described in Appendix D
    of said paper.

    :param av: accelerating voltage/V (not relativistically corrected), num
    :param m_list: list of m-values of aberrations, list[ints]
    :param n_list: list of n-values of aberrations, list[ints]
    :param cmn_list: list of magnitudes/m of aberrations, list[num]
    :param theta_cmn_list: list of angles/rad associated with aberrations in Table A.1, list[num]
    :param alpha_tilt_angle_max: maximum tilt angle/rad mentioned by Sawada et al., 2008, num
    :param h: higher-differential term in Equation 9b (dimensionless), num
    # todo: check units of h (minor todo)
    :param imdim: output Ronchigram has imdim x imdim elements, int

    :return CTF: contrast transfer function (dimensionless), torch.Tensor
    # todo: make sure it is dimensionless (minor todo)
    """
    # The below alpha_tilt_angle calculations were written with the assistance of (Schnitzer, 2020)

    # 1D vector containing convergence angles' possible x- and y- component values. The third argument is
    # imdim because the images in question contain imdim by imdim pixels.
    al_components = np.linspace(-alpha_tilt_angle_max, alpha_tilt_angle_max, imdim)

    # 2d grid of convergence angle vectors formed of different combinations of x- and y- components above.
    al_grid = np.meshgrid(al_components, al_components)

    al_xx, al_yy = al_grid

    # Array of alpha tilt angles at each grid point
    alpha_tilt_angle = np.sqrt(al_xx ** 2 + al_yy ** 2)
    # In C:\Users\james\PycharmProjects\Ronchigram_Simulations\Simulation_3.py, this is
    # called al_rr rather than alpha_tilt_angle, and it's called al_rr in (Schnitzer, 2020)
    # todo: make sure it is the same as the alpha in the paper by Sawade et al., 2008

    # Array of azimuthal angle at each grid point, using arctan2 to account for quadrant position.
    theta_azimuth = np.arctan2(al_yy, al_xx)

    # Array of omega values expressed in terms of tilt angle (alpha) and azimuth (azim) of incident ray.
    omega = alpha_tilt_angle * np.exp(-1j * theta_azimuth)
    omega = torch.from_numpy(omega).to(device)
    print(f"omega's elements' type is {omega.dtype}")

    wavelength = calculate_wavelength(av)

    chi_odd = odd_part_chi_abberation_phase_function(wavelength, m_list, n_list, cmn_list, theta_cmn_list, omega)
    chi_even = even_part_chi_abberation_phase_function(wavelength, m_list, n_list, cmn_list, theta_cmn_list, omega)

    # fourier_transform_operand = np.exp(1j * (chi_odd + h)) * 2 * np.sin(chi_even + h)
    fourier_transform_operand = (torch.exp(1j * (chi_odd + h)) * 2 * np.sin(chi_even + h)).to(device)
    # todo: make sure you've copied this equation correctly, given the ambiguity of whether the sin is
    #   in the exponentiation or not in Equation 9b of (Sawada et al., 2008)

    # todo: now, I assume that numpy.fft.fft2 matches Appendix C of the paper by
    #   Sawada et al., 2008. I will go for it for now but I must make sure. Could
    #   even use the equation in Appendix C directly
    # todo: make sure the below shouldn't be multiplied by G (I don't think Equation 9b and Appendix C imply it should)
    CTF = fft2(fourier_transform_operand)
    # TODO: Here, CTF's complex part may not really be 0, so I will do CTF.real and see if a Ronchigram forms
    CTF = CTF.real

    return CTF

def calculate_inter_param(av):
    """Returns the interaction parameter via equations given in
    https://www.ccp4.ac.uk/ccp4-ed/readings/Kirkland2010/#:~:text=The%20specimen%20is%20thin%20enough,y%2Cz)dz,
    (Schnitzer, Sung and Hovden, 2019) and https://www.globalsino.com/EM/page4217.html. The first two of these give an
    equation for interaction parameter as a function of m, where m is relativistic electron mass. I have derived this
    in my lab book in the 26/10/2021 entry. When subbed into the equation from the first two references, I get Equation
    4217b from the third reference. The first reference's equation seems to match Equation 9b from Sawada et al., 2008,
    which is why I use it for now. Requires numpy to be imported as np and scipy.constants as sc.

    :param av: accelerating voltage/V (not relativistically corrected), num

    :return inter_param: interaction parameter/C.s**2.kg**-1.m**-3, num
    """
    # todo: make sure of the above units and that this is the correct equation

    wavelength = calculate_wavelength(av)

    # The below splitting more directly corresponds to the third reference's equation
    inter_param_lhs = 2 * np.pi * sc.m_e * sc.e * wavelength / sc.h**2
    inter_param_sqrt = np.sqrt(1 + sc.h**2 / (sc.m_e**2 * sc.c**2 * wavelength**2))
    inter_param = inter_param_lhs * inter_param_sqrt

    return inter_param

def calculate_phi_G(av, imdim):
    """Only formulation found for thin specimen projected potential is one mentioned in (Schnitzer, Sung and Hovden,
    2019). They intend to simulate white noise values between 0 and the inverse of (interaction parameter for an
    accelerating voltage of 200kV); in shifted_ronchigram.m, they use accelerating voltage 300kV for simulations, and
    inverse here is that for 300kV. I use inverse for accelerating voltage simulations are for. Will end up with unity
    (inter_param(av)/inter_param(av)), but for reasons in my lab book (26/11/2021), sounds viable.

    :param wavelength: relativistically-corrected electron wavelength/m, num
    :param imdim: output Ronchigram has imdim x imdim elements, int

    :return noise_fun: thin specimen projected potential, C**-1.s**-2.kg.m**3, torch.Tensor
    # todo: make sure of units above
    """
    # todo: not sure if phi_G here is explicity in terms of G as it is in Equation 9b, make sure this is okay
    # todo: figure out how to implement the kmax/2 bit of Schnitzer
    # Just going to copy my code from Ronchigram_Simulations/Simulation_3.py, which is translated from MATLAB code from
    # (Schnitzer, 2020).

    nnoise = 1  # Presumably the number of noise functions
    noisefact = 16  # Particularly have no idea about this

    assert imdim % noisefact == 0
    noise_kernel_size = int(imdim/noisefact)    # 256 ->32, 512 ~ 32, 1024 -> 128   # Comment meaning

    assert imdim % noise_kernel_size == 0
    resize_factor = int(imdim/noise_kernel_size)

    noise_fn = np.zeros((noise_kernel_size, noise_kernel_size))
    for i in range(nnoise):
        noise_fn += standard_normal(size=(noise_kernel_size, noise_kernel_size))

    noise_fn /= nnoise
    noise_fn += 1
    noise_fn /= 2

    # todo: ask Chen if it is sufficient to approximate the imresize process in (Schnitzer, 2020c) using the nearly
    #   equivalent Python code below, despite it leading to a slightly different array - presumably this is okay since
    #   noise_fun is only an approximation of specimen potential, but should I do better to replicate their MATLAB code
    #   here exactly?
    new_shape = tuple([resize_factor * i for i in noise_fn.shape])
    noise_fn = np.array(Image.fromarray(noise_fn).resize(size=new_shape))

    white_noise_value_min = 0
    white_noise_value_max = 1 / calculate_inter_param(av)

    noise_fn *= white_noise_value_max

    noise_fn = torch.from_numpy(noise_fn).to(device)
    # TODO: bit concerned about the size here, especially since convolution will be done
    # noise_fn.unsqueeze_(0)
    # print(f"phi_G's size is {noise_fn.size()}")

    return noise_fn

def calculate_Ronchigram(av, m_list, n_list, cmn_list, theta_cmn_list, h, alpha_tilt_angle_max, imdim):
    """Calculates the Ronchigram for the above parameters. Requires convolve2d to be imported from scipy.signal.

    :param av: accelerating voltage/V (not relativistically corrected), num
    :param m_list: list of m-values of aberrations, list[ints]
    :param n_list: list of n-values of aberrations, list[ints]
    :param cmn_list: list of magnitudes/m of aberrations, list[num]
    :param theta_cmn_list: list of angles/rad associated with aberrations in Table A.1, list[num]
    :param h: higher-differential term in Equation 9b (dimensionless), num
    :param inter_param: interaction parameter/C.s**2.kg**-1.m**-3, num
    :param phi_G: thin specimen projected potential, C**-1.s**-2.kg.m**3, num
    :param alpha_tilt_angle_max: maximum tilt angle/rad mentioned by Sawada et al., 2008, num
    :param imdim: output Ronchigram has imdim x imdim elements, int

    :return ronch: un-normalised Ronchigram array, torch.Tensor
    """

    CTF = calculate_CTF(av, m_list, n_list, cmn_list, theta_cmn_list, alpha_tilt_angle_max, imdim, h)
    # CTF.unsqueeze_(0)
    # print(CTF.size())

    inter_param = calculate_inter_param(av)
    phi_G = calculate_phi_G(av, imdim)

    # todo: make sure to do the convolution in Equation 9b as it is explained in Equation A.4
    #   of the paper by Sawada et al., 2008
    # I'm going to assume for now that this convolution can be done using scipy.signal.convolve2d
    # (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html)
    # Going to use mode="same" for now
    # todo: I am assuming inter_param should not be part of the convolution, but I must make sure
    # todo: play with other parameters of convolve2d if applicable
    # ronch = 1 + inter_param * convolve2d(in1=phi_G, in2=CTF, mode="same")
    # ronch = 1 + inter_param * conv2d(input = CTF, weight = phi_G, padding="same")
    ronch = 1 + inter_param * phi_G * CTF
    # TODO: since phi_G and CTF are the same size I simply multiplied them - check if this was the right decision

    return ronch

def normalise_ronchigram(ronch):
    """ Takes a Ronchigram and normalises it in accordance with the normalisation in (Schnitzer, 2020). Requires numpy
    to be imported as np.

    :param ronch: un-normalised Ronchigram, numpy.ndarray

    :return ronch: normalised Ronchigram, numpy.ndarray
    """

    ronch -= torch.amin(ronch).to(device)
    ronch /= torch.amax(ronch).to(device)

    return ronch

def poisson_noise_ronch(ronch, PEAK):
    """ Takes a normalised Ronchigram and implements Poisson noise in it. One of the parameters for this Poisson noise
    is PEAK, which gives an indication of the amount of Poisson noise present. Requires numpy to be imported np.

    :param ronch: Ronchigram, numpy.ndarray
    :param PEAK: Poisson noise parameter, num

    :return poisson_noisy_ronc: Poisson-noisy Ronchigram, numpy.ndarray
    """

    # https://stackoverflow.com/questions/19289470/adding-poisson-noise-to-an-image for assistance, except with 255
    #   replace by 1 because normalisation of ronch somewhere above leads to array elements between 0 and 1. I am taking
    #   yuxiang.li's suggestion, but only his upper one, because the lower one looks like it simply adds the Poisson noise
    #   to the image, which doesn't seem right since Poisson noise isn't additive.
    # rng is random number generator, see https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.poisson.html
    
    # rng = np.random.default_rng()
    # poisson_noisy_ronch = rng.poisson(ronch * PEAK) / PEAK
    poisson_noisy_ronch = (torch.poisson(ronch * PEAK) / PEAK).to(device)

    return poisson_noisy_ronch

# Maximum alpha (tilt angle)/rad, currently using simdim from Ronchigram_Simulations/Simulation_3.py
alpha_tilt_angle_max = 100 * 10**-3
# EfficientNet-B7 has a resolution of 600 (see CNN_5.py in My_CNNs of VS Code on group computer) but going to have
#   imdim as 1024 for now so it works with noisefact in the noise function calculation.
# todo: change noisefact so it works with 600 resolution, or resize output Ronchigram to 600
imdim = 1024
un_norm_ronch = calculate_Ronchigram(av, m_list, n_list, cmn_list, theta_cmn_list, h, alpha_tilt_angle_max, imdim).to(device)
print(un_norm_ronch)
print("Un-normalised Ronchigram simulated")

norm_ronch = normalise_ronchigram(un_norm_ronch).to(device)
print(norm_ronch)
print("Ronchigram normalised")

PEAK = 100
noisy_ronch = poisson_noise_ronch(norm_ronch, PEAK).to(device)
print("Poisson noise added to Ronchigram")

plt.imshow(noisy_ronch.cpu(), cmap="gray", interpolation="nearest")
plt.show()

print("Finished")

# REFERENCES
# todo: some of the information from Sawada et al. 2008 is cited from other papers,
#   so make sure you have done any secondary citations correctly

# Sawada, H. et al. (2008) “Measurement method of aberration from Ronchigram by autocorrelation function,”
# Ultramicroscopy, 108(11), pp. 1467–1475. doi:10.1016/J.ULTRAMIC.2008.04.095.

# Schnitzer, N. (2020) ronchigram-matlab/shifted_ronchigram.m at master · noahschnitzer/ronchigram-matlab. Available
# at: https://github.com/noahschnitzer/ronchigram-matlab/blob/master/simulation/shifted_ronchigram.m (Accessed: October
# 21, 2021).

# Schnitzer, N., Sung, S.H. and Hovden, R. (2019) “Introduction to the Ronchigram and its Calculation with
# Ronchigram.com,” Microscopy Today, 27(3), pp. 12–15. doi:10.1017/S1551929519000427.

# Schnitzer, N., Sung, S.H. and Hovden, R. (2020) “Optimal STEM Convergence Angle Selection Using a Convolutional
# Neural Network and the Strehl Ratio,” Microscopy and Microanalysis, 26(5). doi:10.1017/S1431927620001841.