# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


import copy
import dnnlib
from dnnlib import EasyDict
import config

#----------------------------------------------------------------------------

if 1:
    desc          = 'sgan'                                                                 # 包含在结果子目录名称中的描述字符串。
    train         = EasyDict(run_func_name='training.training_loop.training_loop')         # 训练过程设置。
    G             = EasyDict(func_name='training.networks_stylegan.G_style')               # 生成网络架构设置。
    D             = EasyDict(func_name='training.networks_stylegan.D_basic')               # 判别网络架构设置。
    G_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # 生成网络优化器设置。
    D_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # 判别网络优化器设置。
    G_loss        = EasyDict(func_name='training.loss.G_wgan')                             # 生成损失设置。
    D_loss        = EasyDict(func_name='training.loss.D_wgan_gp')                          # 判别损失设置。
    dataset       = EasyDict()                                                             # 数据集设置，在后文确认。
    sched         = EasyDict()                                                             # 训练计划设置，在后文确认。
    grid          = EasyDict(size='4k', layout='random')                                   # setup_snapshot_image_grid()相关设置。
    submit_config = dnnlib.SubmitConfig()                                                  # dnnlib.submit_run()相关设置。
    tf_config     = {'rnd.np_random_seed': 1000}                                           # tflib.init_tf()相关设置。

    # 数据集。
    desc += '-berea96';dataset = EasyDict(tfrecord_dir='berea', resolution=64)

    # GPU数量。
    desc += '-2gpu';
    submit_config.num_gpus = 2;sched.minibatch_base = 4;sched.minibatch_dict = {2: 128, 4: 64, 8: 32, 16: 16, 32: 8}    # ‘2’ corresponds to the minimum resolution of 6, ‘32’ corresponds to the maximum resolution of 96

    # 默认设置。
    train.total_kimg = 500
    sched.lod_initial_resolution = 2                                                                                    # ‘2’ corresponds to the minimum resolution of 6, ‘32’ corresponds to the maximum resolution of 96
    sched.G_lrate_dict = {2: 0.002, 4: 0.002, 8: 0.002, 16: 0.002, 32: 0.002}
    sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)

#----------------------------------------------------------------------------

def main():
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
