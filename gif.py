import imageio
from glob import glob
from tqdm import tqdm

filenames = glob('car\\*')

with imageio.get_writer('gan\\training.gif', mode='I',duration=0.01) as writer:
    for filename in tqdm(filenames):
        image = imageio.imread(filename)
        writer.append_data(image)
