from matplotlib import pyplot as plt
from thinkdsp import *
from PIL import Image
import pandas as pd
import urllib.request


class ImageCatalog():
    def __init__(self, image_csv_path):
        self.image_csv_path = image_csv_path
        self._get_metadata()

    def _get_metadata(self):
        cols = ['ImageName', 'FileDir', 'CollectDate', 'Instrument']  # columns in image list csv
        df = pd.read_csv(self.image_csv_path, header=0, usecols=cols, skip_blank_lines=True,
                         encoding='utf-8')  # 'latin1'
        self.image_names = df['ImageName'].tolist()
        self.image_urls = df['FileDir'].tolist()

    def get_image_path(self, image_index):
        image_url = self.image_urls[image_index]
        image_name = self.image_names[image_index]
        if '.jpg' in image_url:
            image_format = '.jpg'
        elif '.png' in image_url:
            image_format = '.png'
        image_path = './images/' + image_name + image_format
        urllib.request.urlretrieve(image_url, image_path)
        return image_path

    def get_image_name(self, image_index):
        return self.image_names[image_index]


class Sonification():
    def __init__(self, image_path, freqs, sonif_duration):
        self.image_path = image_path
        self.song = 22050
        self.freqs = freqs
        self.sonif_duration = sonif_duration
        self._get_pixels()
        self._make_sonification()

    def _get_pixels(self):
        img = Image.open(self.image_path)
        img = boost_contrast(img)

        imgR = img.resize(size=(img.width, len(self.freqs)), resample=Image.LANCZOS)
        self.pixels = np.array(imgR.convert('L')) / 255  # normalize

    def _make_sonification(self):
        self.wave = additive_synth(self.pixels, self.freqs, self.song, self.sonif_duration)
        self.wave.normalize(0.9)
        self.y = self.wave.ys

    def save_sonification(self, path):
        self.wave.write(path)


def boost_contrast(image):
    """boost contrast in current image with cosine curve

    argument:PIL image (RGB)

    returns: PIL image (RGB)
    """

    im_array = np.array(image)
    im_array = 255. / 2 * (1 - np.cos(np.pi * im_array / np.amax(im_array)))
    # im_array = map_value(-np.cos(np.pi*im_array/np.amax(im_array)), -1, 1, 0, 255)
    image = Image.fromarray(im_array.astype(np.uint8), "RGB")  # .show()
    return image


def additive_synth(pixel_array, freqs, fs, duration):
    """Converts array to wave with additive synthesis

    returns: thinkdsp wave
    """
    height, width = np.shape(pixel_array)
    freqs_rev = np.array(freqs[::-1])

    Ns = int(fs * duration)

    lut = LUT()  # create sine wave look up table
    delPhase = lut.N * freqs_rev / fs  # phase increment for each frequency
    np.random.seed(0)
    phi = lut.N * np.random.rand(np.shape(pixel_array)[0])  # set initial phase for each frequency

    ts = np.linspace(0, duration, Ns, endpoint=False)

    ys = []
    for n in range(Ns):
        colindex = int((n / Ns * width))  # or ceil or floor?
        colindexRem = n / Ns * width - colindex

        phi = (phi + delPhase) % lut.N  # find new phase of each frequency component
        phi_int = phi.astype(int)

        Amp = pixel_array[:, colindex] + colindexRem * (
                    pixel_array[:, min(colindex + 1, width - 1)] - pixel_array[:, colindex])

        yi = Amp * lut.waveLUT[phi_int]
        ys.append(yi.sum())

    wave = Wave(ys, ts, framerate=fs)
    spectrum = wave.make_spectrum()
    spectrum.low_pass(cutoff=max(freqs), factor=0.)  # only if artifact
    wave = spectrum.make_wave()
    return wave


class LUT:
    '''look up table object'''

    def __init__(self, waveform='sine', M=1, N=2 ** 10):
        self.N = N
        if isinstance(waveform, str):
            if waveform not in ['sin', 'sine', 'cos', 'square', 'tri', 'triangle', 'saw', 'sawtooth']:
                print('Waveform name must be one of', ['sin', 'sine', 'cos', 'square', 'triangle', 'sawtooth'])
            self.waveform = waveform
            self.M = M
            self._make_wave()
        if isinstance(waveform, list) or isinstance(waveform, np.ndarray):
            self.M = None
            self.custom(waveform)

    def _make_wave(self):
        lutSamp = np.arange(self.N)
        if self.waveform in ['sin', 'sine']:
            self.waveform = 'sine'
            self.waveLUT = np.sin(2 * np.pi * lutSamp / self.N)
        if self.waveform == 'cos':
            self.waveLUT = np.cos(2 * np.pi * lutSamp / self.N)
        if self.waveform == 'square':
            self.waveLUT = np.zeros(self.N)
            for m in range(self.M):
                self.waveLUT += 4 / np.pi / (2 * m + 1) * np.sin((2 * m + 1) * 2 * np.pi * lutSamp / self.N)
        if self.waveform in ['tri', 'triangle']:
            self.waveform = 'triangle'
            self.waveLUT = np.zeros(self.N)
            for m in range(self.M):
                self.waveLUT += 8 / (np.pi) ** 2 * (-1) ** m / (2 * m + 1) ** 2 * np.sin(
                    (2 * m + 1) * 2 * np.pi * lutSamp / self.N)
        if self.waveform in ['saw', 'sawtooth']:
            self.waveLUT = np.zeros(self.N)
            for m in range(self.M):
                self.waveLUT += -1 / np.pi / (m + 1) * np.sin((m + 1) * 2 * np.pi * lutSamp / self.N)

    def custom(self, wave_samples):
        self.waveform = 'custom'
        wave_samples = map_value(wave_samples, min(wave_samples), max(wave_samples), -1, 1.)
        ts_input = np.linspace(0, self.N, len(wave_samples))  # is this right? endpoint=True?
        ts_target = np.arange(0, self.N)
        self.waveLUT = np.interp(ts_target, ts_input, wave_samples, left=None, right=None, period=None)
        return self

    def plot(self):
        plt.figure(figsize=(10, 3))

        plt.plot(np.arange(0, self.N), self.waveLUT)
        plt.title(self.waveform + ', M =' + str(self.M))
        plt.show()