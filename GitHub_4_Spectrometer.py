import scipy.io as sio
import matplotlib.pyplot as plt
from minispec import Minispec, find_devices
import time
from PIL import Image

global curr_image
curr_image = 20

# Function to load image based on given number
def load_image(num):
    # Directory containing image overlays
    dir_overlays = 'C:\\Users\\Lenovo ThinkPad\\OneDrive - University of Surrey\\Desktop\\dissertation vscode\\SAM_dataset' #add the correct path
    # Construct image path
    image_path = "{}/image_{}/overlays/image_{}_overlay_{}.jpg".format(dir_overlays, curr_image, curr_image, num) #add the correct path
    print(image_path)
    # Open and return the image
    return Image.open(image_path)

# Find connected spectrometers
spectrometers = find_devices(find_first=True, sock_timeout=50, search_timeout=50)
print(spectrometers)

print("Found. {} spectrometer(s).".format(len(spectrometers)))
n_saved = 0

# Connect to the first found spectrometer
if len(spectrometers) > 0:
    (hostname, port), iface, serial = spectrometers.pop()
    print("Connecting to {}, via {}".format(hostname, iface.decode()))

    # Create figure and subplots for image and spectrum
    figure1 = plt.figure("Overlay")
    ax1 = figure1.add_subplot(111)
    figure2 = plt.figure("Spectrum")
    ax2 = figure2.add_subplot(111)

    with Minispec(hostname) as mspec:
        print("hello")
        mspec.exposure = 100
        print("Exposure set to {} ms.".format(mspec.exposure))
        print("Current calibration {}".format(mspec.calibration))

        wavelengths, spectrum = mspec.wavelengths, mspec.spectrum()
        # mspec.dark = spectrum
        save_spectrum = False
        nan_out = False

        # Function to handle key press events
        def on_key(event):
            global save_spectrum
            global nan_out
            if event.key == 'p':
                save_spectrum = True
            if event.key == 'n':
                nan_out = True

        # Connect key press event to function
        figure2.canvas.mpl_connect('key_press_event', on_key)

        # Load initial image
        image = load_image(n_saved)
        ax1.imshow(image)
        ax1.set_title('Overlay')
        ax1.axis('off')

        while True:
            wavelengths, spectrum = mspec.wavelengths, mspec.spectrum()

            # Clear and plot spectrum
            ax2.cla()
            ax2.plot(wavelengths, spectrum)
            ax2.set_title('Spectrum')
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Counts')

            # Pause to allow plot updates
            plt.figure("Overlay")
            plt.pause(0.01)
            plt.figure("Spectrum")
            plt.pause(0.01)

            # Handle saving spectrum with NaN values
            if nan_out:
                filename = f'image_{curr_image}/spectrum' + str(n_saved) + '.mat'
                spectrum = "N/A"
                sio.savemat(filename, {'wavelengths': wavelengths, 'spectrum': spectrum})
                n_saved += 1
                nan_out = False
                print("NAN Spectrum saved as", filename)
                image = load_image(n_saved)
                ax1.imshow(image)
                ax1.set_title('Overlay')
                ax1.axis('off')

            # Handle saving real spectrum
            if save_spectrum:
                # Save the spectrum as a .mat file
                filename = f'image_{curr_image}/spectrum' + str(n_saved) + '.mat'
                sio.savemat(filename, {'wavelengths': wavelengths, 'spectrum': spectrum})
                print("REAL Spectrum saved as", filename)
                save_spectrum = False
                n_saved += 1
                image = load_image(n_saved)
                ax1.imshow(image)
                ax1.set_title('Overlay')
                ax1.axis('off')
