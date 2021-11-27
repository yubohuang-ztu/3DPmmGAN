# Thanks to StyleGAN2 provider —— Copyright (c) 2019, NVIDIA CORPORATION.
# This work is trained by Copyright(c) 2018, seeprettyface.com, BUPT_GWY.
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import pickle
import os
import h5py

_cached_networks = dict()

def load_networks(path):
    stream = open(path, 'rb')
    tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')
    _cached_networks[path] = G, D, Gs
    return G, D, Gs

def generate_images(network_pkl, coeffs, direction_file,noise_seed):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = load_networks(network_pkl)
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    np.random.seed(noise_seed)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})

    fixed_dlatent_direction = np.load('dlatent_directions/' + direction_file)
    origin = fixed_dlatent_direction[0][np.newaxis, ...]
    direction = fixed_dlatent_direction[1][np.newaxis, ...]

    for i, coeff in enumerate(coeffs):
        dlatent_vector = origin.copy()
        dlatent_vector = origin + coeff * direction
        images = Gs.components.synthesis.run(dlatent_vector, **Gs_kwargs)
        h5_data = images[0]
        h5_data = h5_data.squeeze()
        with h5py.File('results/' + 'fake' + str(i) + '.hdf5', 'w') as f:
            f.create_dataset('data', data=h5_data, dtype="i8", compression="gzip")

def main():
    os.makedirs('dlatent_directions/', exist_ok=True)
    os.makedirs('networks/', exist_ok=True)
    os.makedirs('results/', exist_ok=True)

    # path to model
    network_pkl = 'networks/aggregation.pkl'

    # example direction vector, choose one of them (Control the direction of morphological change)
    '''
        aggregation_direction_x.npy
        aggregation_direction_y.npy
        aggregation_direction_z.npy
        aggregation_direction_v.npy
    '''
    direction_file = 'aggregation_direction_x.npy'
    coeffs = [0., 0.25, 0.5, 0.75, 1.0]

    # seed of noise_vars (Control morphology)
    noise_seed = 1000

    generate_images(network_pkl, coeffs, direction_file,noise_seed)

if __name__ == "__main__":
    main()
