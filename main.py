# Created by Tyler Lovett

import numpy as np  # Imports numpy to use the cos/sin along with FTT/IFFT commands
from spellchecker import SpellChecker   # To be used for slight error correction

# This function just loads the input.txt as is
def load_signal(filename):
    return np.loadtxt(filename)

# Loads the incoming preamble
def load_preamble(filename):
    data = []  # Creates an empty list to store the updated preamble.txt
    with open(filename, "r") as f:
        for line in f:
            # Python doesn't view i as imaginary so j is used instead
            # This ensures that the numbers with i are now imaginary numbers
            line = line.strip().replace("i", "j")
            data.append(complex(line))
    return np.array(data)

# The function creates the diagram of the 16 QAM
# The function demodulates the 16-QAM symbols into bits
def qam_16(symbols):
    # Defines the 16-QAM constellation levels and bit map
    level = np.array([-3, -1, 1, 3])
    bit_map = {-3: "10", -1: "11", 1: "01", 3: "00"}
    bits = ""

    # Demodulates each symbol
    for sym in symbols:
        # Separates the combined symbol into real (I) and imaginary (Q) components
        I = np.real(sym)
        Q = np.imag(sym)

        # Find the nearest constellation level for I/Q
        I_hat = near_level(I, level)
        Q_hat = near_level(Q, level)

        # Places bits together to form a full byte in ASCII
        bits += bit_map[Q_hat] + bit_map[I_hat]
    return bits

# This function will find the nearest level related to the 16 QAM
def near_level(symbols, levels):
    # Determines the closest level for the specific symbol
    return levels[np.argmin(np.abs(levels - symbols))]

# This function decodes the ASCII into text
def decode(ASCII):
    message = ""
    # Loops through the entire ASCII between every 8 bits
    for i in range(0, len(ASCII), 8):
        byte = ASCII[i:i + 8]
        # If a byte is less than 8 bits then it's discarded
        if (len(byte) < 8):
            break
        # Converts the ASCII into a readable letter/character
        message += chr(int(byte, 2))
    return message

# The main function that will run all
def main():
    # Sets the parameters
    fc = 20  # center frequency in Hz
    N = 3000  # number of samples
    fs = 100  # sampling rate in Hz
    symbol_rate = 10    # symbol rate in Hz

    # TODO: Change the file names to the file you want to decode
    input_signal = load_signal("input.txt")     # Modify this line for the input file
    preamble = load_preamble("preamble.txt")    # Modify this line for the preamble file

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DOWNCONVERSION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n = np.arange(N)
    T = n / fs  # Calculates the sampling period

    # Computes the cos and sin functions
    cos_carrier = np.cos(2 * np.pi * fc * T)
    sin_carrier = np.sin(2 * np.pi * fc * T)

    # Multiplies the input signal with the cos/sin functions
    # This converts I/Q down to baseband
    I = input_signal * cos_carrier
    Q = input_signal * sin_carrier

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FILTER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Computes the Fast Fourier Transform
    FFT_I = np.fft.fft(I)
    FFT_Q = np.fft.fft(Q)

    # Calculates the frequencies based off the 3000 long input signal
    # The fftfreq recursively calculates the frequency for each element
    frequencies = np.fft.fftfreq(N, 1/fs)

    # Sets the frequency range
    freq_range = (frequencies >= -5.1) & (frequencies <= 5.1)

    # Filters the frequencies outside the range from the I/Q to be 0
    filter_I = FFT_I * freq_range
    filter_Q = FFT_Q * freq_range

    # Eliminates high frequencies retains the baseband form
    IFFT_I = np.fft.ifft(filter_I)
    IFFT_Q = np.fft.ifft(filter_Q)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DOWNSAMPLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    samples_per_symbols = fs // symbol_rate  # Takes every 10th sample

    # The 10th sample is taken based off the IFFT
    # Each downsample has 300 elements
    downsample_I = IFFT_I[::samples_per_symbols]
    downsample_Q = IFFT_Q[::samples_per_symbols]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CORRELATE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sets the downsampled I/Q in the baseband form
    baseband = downsample_I + 1j * downsample_Q

    # Performs the correlation using the baseband and the preamble
    correlation = np.abs(np.correlate(baseband, preamble, mode="valid"))

    # Locates the start of the preamble based off the largest value
    start_index = np.argmax(correlation)

    # Extracts the symbols
    symbols = baseband[start_index + len(preamble):]

    # The signal gets normalized as it was amplified in the previous steps
    scale = np.mean(np.abs(symbols))
    symbols = (symbols / scale) * 3

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEMODULATE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Calls 16-QAM to demodulate the symbols
    ASCII = qam_16(symbols)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ASCII TO TEXT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Decodes ASCII into text based from the demodulate step
    decode_mess = decode(ASCII)
    print(decode_mess)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ERROR CORRECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # This applies a simple spellchecker on the decoded message
    # A spellchecker will fix some of the noise but not all
    # Any capitals or wrong punctuation doesn't get fixed
    # The errors are off by 1 bit
    # A bit flip would be needed to be implemented for a proper spellcheck

    """
    spell = SpellChecker()
    # Splits the message into the individual words to correct each word
    split = decode_mess.split()
    correct_mess = ""

    # For every word it corrects it and adds to the correct message string
    for word in split:
        correct_word = spell.correction(word)
        # This handles words that can't be corrected and just returns the original
        if correct_word is None:
            correct_word = word
        correct_mess += correct_word + " "
    print(correct_mess)
    """
main()