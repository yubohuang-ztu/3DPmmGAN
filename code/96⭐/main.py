# Thanks to StyleGAN2 provider —— Copyright (c) 2019, NVIDIA CORPORATION.
# This work is trained by Copyright(c) 2018, seeprettyface.com, BUPT_GWY.
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import pickle
import os

_cached_networks = dict()

def load_networks(path):
    stream = open(path, 'rb')
    tflib.init_tf()
    with stream:
        G, D, Gs = pickle.load(stream, encoding='latin1')
    _cached_networks[path] = G, D, Gs
    return G, D, Gs

def generate_images(network_pkl, AdaIN_Channel_fixed, Noise_Channel_fixed, generated_num, phase_trans = True):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = load_networks(network_pkl)
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False

    for i in range(generated_num):
        if Noise_Channel_fixed:
            np.random.seed(0)
        else:
            np.random.seed(i)
        noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
        tflib.set_vars({var: np.random.randn(*var.shape.as_list()) for var in noise_vars})

        if AdaIN_Channel_fixed:
            rnd = np.random.RandomState(0)
        else:
            rnd = np.random.RandomState(i)
        z = rnd.randn(1, *Gs.input_shape[1:])
        dlatent = Gs.components.mapping.run(z, None)
        images = Gs.components.synthesis.run(dlatent, **Gs_kwargs)
        tflib.save_img(images, i, phase_trans)


def main():
    os.makedirs('results/', exist_ok=True)

    # path to model
    network_pkl =

    # AdaIN Channels fixed?
    AdaIN_Channel_fixed = False

    # Noise Channels fixed?
    Noise_Channel_fixed = False

    # Number of generated images
    generated_num = 20

    generate_images(network_pkl, AdaIN_Channel_fixed, Noise_Channel_fixed, generated_num)

if __name__ == "__main__":
    main()
