from resource import *


# By passing in an image file name that is in the images directory (jpg),
# create_sonification produces a wav file where that image is transformed into
# sound, or better known as sonification. This wav file is saved in sonifications directory
def create_sonification(image_file):

    # set tempo and frequencies for wav file
    tempo = 103
    freqs = [195.99771799087466, 220.0, 246.94165062806206, 293.6647679174076, 329.6275569128699, 391.99543598174927,
             440.0, 493.8833012561241, 587.3295358348151, 659.2551138257398, 783.9908719634985, 880.0, 987.766602512248,
             1174.65907166963, 1318.5102276514797, 1567.981743926997]

    image_filename = image_file  # name of jpg file
    image_path = './images/' + image_filename + '.jpg'

    # set beats per minute and duration
    beats_per_bar = 4
    time_per_bar = beats_per_bar * 60 / tempo
    n_bars = 16
    sonif_duration = n_bars * time_per_bar

    # using the classes in the resource file, sonify the image and save in sonifications directory
    sonification = Sonification(image_path, freqs, sonif_duration)
    sonification.save_sonification('./sonifications/' + image_file + '.wav')


if __name__ == '__main__':
    create_sonification("carina_nircam_g")
