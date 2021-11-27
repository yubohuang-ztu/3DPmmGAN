# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

import config
import train
from training import dataset
from training import misc
import h5py


#----------------------------------------------------------------------------

def process_reals(x, lod, mirror_augment, drange_data, drange_net):

    with tf.name_scope('ProcessReals'):
        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
            #x = (x / 255 - 0.5) / 0.5
        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3], s[4]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))

        with tf.name_scope('FadeLOD'):
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2, s[4]//2, 2])
            # 8个48*48*48的区域，图片区域化，使用均值像素代替
            y = tf.reduce_mean(y, axis=[3, 5, 7], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3], s[4]])
            x = tflib.lerp(x, y, lod - tf.floor(lod))
        with tf.name_scope('UpscaleLOD'):  # 缩放以匹配网络的预期输入/输出大小。
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1, s[4], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor, s[4] * factor])
        return x

#----------------------------------------------------------------------------

def training_schedule(
    cur_nimg,
    training_set,
    num_gpus,
    lod_initial_resolution  = 4,        # 开始训练时使用的图像分辨率。
    lod_training_kimg       = 50,       # 在将分辨率提高一倍之前要显示数千个真实图像。
    lod_transition_kimg     = 50,       # 淡入新图层时要显示数千个真实图像。
    minibatch_base          = 16,       # 小批量大小的最大值，在GPU之间平均分配。
    minibatch_dict          = {},       # 特定分辨率的重写。
    max_minibatch_per_gpu   = {},       # 每个GPU的特定分辨率的小批量大小的最大值。
    G_lrate_base            = 0.01,     # 生成器的学习率。
    G_lrate_dict            = {},       # 特定分辨率的重写。
    D_lrate_base            = 0.01,     # 判别器的学习率。
    D_lrate_dict            = {},       # 特定分辨率的重写。
    lrate_rampup_kimg       = 0,        # 学习率提升的持续时间。
    tick_kimg_base          = 160,      # 训练过程中生成快照的默认间隔。
    tick_kimg_dict          = {4: 160, 8:140, 16:120, 32:100, 64:80}):  # 特定分辨率的重写。

    # Initialize result dict.
    s = dnnlib.EasyDict()
    s.kimg = cur_nimg / 1000.0

    # Training phase.
    phase_dur = lod_training_kimg + lod_transition_kimg
    phase_idx = int(np.floor(s.kimg / phase_dur)) if phase_dur > 0 else 0
    phase_kimg = s.kimg - phase_idx * phase_dur

    # Level-of-detail and resolution.
    s.lod = training_set.resolution_log2
    s.lod -= np.floor(np.log2(lod_initial_resolution))
    s.lod -= phase_idx
    if lod_transition_kimg > 0:
        s.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
    s.lod = max(s.lod, 0.0)
    s.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(s.lod)))

    # Minibatch size.
    s.minibatch = minibatch_dict.get(s.resolution, minibatch_base)
    s.minibatch -= s.minibatch % num_gpus
    if s.resolution in max_minibatch_per_gpu:
        s.minibatch = min(s.minibatch, max_minibatch_per_gpu[s.resolution] * num_gpus)

    # Learning rate.
    s.G_lrate = G_lrate_dict.get(s.resolution, G_lrate_base)
    s.D_lrate = D_lrate_dict.get(s.resolution, D_lrate_base)
    if lrate_rampup_kimg > 0:
        rampup = min(s.kimg / lrate_rampup_kimg, 1.0)
        s.G_lrate *= rampup
        s.D_lrate *= rampup

    # Other parameters.
    s.tick_kimg = tick_kimg_dict.get(s.resolution, tick_kimg_base)
    return s

#----------------------------------------------------------------------------
# Main training script.

def training_loop(
    submit_config,
    G_args                  = {},       # 生成网络的设置。
    D_args                  = {},       # 判别网络的设置。
    G_opt_args              = {},       # 生成网络优化器设置。
    D_opt_args              = {},       # 判别网络优化器设置。
    G_loss_args             = {},       # 生成损失设置。
    D_loss_args             = {},       # 判别损失设置。
    dataset_args            = {},       # 数据集设置。
    sched_args              = {},       # 训练计划设置。
    grid_args               = {},       # setup_snapshot_image_grid()相关设置。
    metric_arg_list         = [],       # 指标方法设置。
    tf_config               = {},       # 随机种子{'rnd.np_random_seed': 1000}
    G_smoothing_kimg        = 10.0,     # 生成器权重的运行平均值的半衰期。
    D_repeats               = 1,        # G每迭代一次训练判别器多少次。
    minibatch_repeats       = 2,        # 调整训练参数前要运行的minibatch的数量。
    reset_opt_for_new_lod   = True,     # 引入新层时是否重置优化器内部状态（例如Adam时刻）？
    total_kimg              = 15000,    # 训练的总长度，以成千上万个真实图像为统计。
    mirror_augment          = False,    # 启用镜像增强？
    drange_net              = [-1,1],   # 将图像数据馈送到网络时使用的动态范围。
    image_snapshot_ticks    = 1,        # 多久导出一次图像快照？
    network_snapshot_ticks  = 2,        # 多久导出一次网络模型存储？
    save_tf_graph           = False,    # 在tfevents文件中包含完整的TensorFlow计算图吗？
    save_weight_histograms  = False,    # 在tfevents文件中包括权重直方图？
    resume_run_id           = None,     # 运行已有ID或载入已有网络pkl以从中恢复训练，None = 从头开始。
    resume_snapshot         = None,     # 要从哪恢复训练的快照的索引，None = 自动检测。
    resume_kimg             = 0.0,      # 在训练开始时给定当前训练进度。影响报告和训练计划。
    resume_time             = 0.0):     # 在训练开始时给定统计时间。影响报告。
    
    # 初始化dnnlib和TensorFlow。
    ctx = dnnlib.RunContext(submit_config, train)
    tflib.init_tf(tf_config)

    # 载入训练集。
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **dataset_args) #verbose=True:表示设置运行的时候显示详细信息
    # 构建网络。
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            G, D, Gs = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            G = tflib.Network('G', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **G_args)
            D = tflib.Network('D', num_channels=training_set.shape[0], resolution=training_set.shape[1], label_size=training_set.label_size, **D_args)
            Gs = G.clone('Gs')
    G.print_layers(); D.print_layers()
    # 构建计算图与优化器
    with tf.name_scope('Inputs'), tf.device('/cpu:0'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        minibatch_split = minibatch_in // submit_config.num_gpus
        Gs_beta         = 0.5 ** tf.div(tf.cast(minibatch_in, tf.float32), G_smoothing_kimg * 1000.0) if G_smoothing_kimg > 0.0 else 0.0

    G_opt = tflib.Optimizer(name='TrainG', learning_rate=lrate_in, **G_opt_args)# 对G进行优化
    D_opt = tflib.Optimizer(name='TrainD', learning_rate=lrate_in, **D_opt_args)# 对D进行优化
    for gpu in range(submit_config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')# 为每个GPU拷贝一份
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')# 为每个GPU拷贝一份
            lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_gpu.find_var('lod'), lod_in)]
            reals, labels = training_set.get_minibatch_tf()# 获得训练数据图片和标签
            reals = process_reals(reals, lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
                G_loss = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=G_opt, training_set=training_set,minibatch_size=minibatch_split, **G_loss_args)
            with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
                D_loss = dnnlib.util.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, training_set=training_set,minibatch_size=minibatch_split, reals=reals, labels=labels, **D_loss_args)
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)  # 注册梯度下降求损失的方法
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)  # 注册梯度下降求损失的方法
    G_train_op = G_opt.apply_updates()  # 反向传播需要的op
    D_train_op = D_opt.apply_updates()  # 反向传播需要的op
    
    # 计算权重平均值（多个gpu）
    Gs_update_op = Gs.setup_as_moving_average_of(G, beta=Gs_beta)
    with tf.device('/gpu:0'):
        try:
            peak_gpu_mem_op = tf.contrib.memory_stats.MaxBytesInUse()
        except tf.errors.NotFoundError:
            peak_gpu_mem_op = tf.constant(0)
    # 设置快照图像网格(已改)
    grid_size, grid_reals, grid_labels, grid_latents = misc.setup_snapshot_image_grid(G, training_set, **grid_args)
    sched = training_schedule(cur_nimg=total_kimg * 1000, training_set=training_set, num_gpus=submit_config.num_gpus, **sched_args)
    grid_fakes = Gs.run(grid_latents, grid_labels, is_validation=True,minibatch_size=sched.minibatch // submit_config.num_gpus)
    # 把图片保存到子项目根目录
    grid_fakes = (grid_fakes * 0.5 + 0.5) * 255
    h5_data = grid_fakes
    h5_data = h5_data.squeeze()
    with h5py.File('results/' + 'fake' + '.hdf5', 'w') as f:
        f.create_dataset('data', data=h5_data, dtype="i8", compression="gzip")
    h5_data1 = grid_reals
    h5_data1 = h5_data1.squeeze()
    with h5py.File('results/' + 'real' + '.hdf5', 'w') as f:
        f.create_dataset('data', data=h5_data1, dtype="i8", compression="gzip")

    # log的收集
    summary_log = tf.summary.FileWriter(submit_config.run_dir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms();
        D.setup_weight_histograms()

    print('Training...\n')
    # 更改训练图片的张数
    ctx.update('', cur_epoch=resume_kimg, max_epoch=total_kimg)
    maintenance_time = ctx.get_last_update_interval()
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0  # 当前训练的tick数从零开始
    tick_start_nimg = cur_nimg
    prev_lod = -1.0
    while cur_nimg < total_kimg * 1000:
        if ctx.should_stop(): break

        # 选择训练参数并配置训练操作。
        sched = training_schedule(cur_nimg=cur_nimg, training_set=training_set, num_gpus=submit_config.num_gpus,**sched_args)
        training_set.configure(sched.minibatch // submit_config.num_gpus, sched.lod)

        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state();
                D_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # 进行训练。
        for _mb_repeat in range(minibatch_repeats):
            for _D_repeat in range(D_repeats):
                tflib.run([D_train_op, Gs_update_op],{lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})  # 出错
                cur_nimg += sched.minibatch
                print('cur_nimg=',cur_nimg)
            tflib.run([G_train_op], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch})

        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 20 or done:
            cur_tick += 1
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = ctx.get_time_since_last_update()
            total_time = ctx.get_time_since_start() + resume_time

            # 报告进度。
            print(
                'tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %-6.1f gpumem %-4.1f' % (
                    autosummary('Progress/tick', cur_tick),
                    autosummary('Progress/kimg', cur_nimg / 1000.0),
                    autosummary('Progress/lod', sched.lod),
                    autosummary('Progress/minibatch', sched.minibatch),
                    dnnlib.util.format_time(autosummary('Timing/total_sec', total_time)),
                    autosummary('Timing/sec_per_tick', tick_time),
                    autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                    autosummary('Timing/maintenance_sec', maintenance_time),  #maintenance_time 持续时间
                    autosummary('Resources/peak_gpu_mem_gb', peak_gpu_mem_op.eval() / 2 ** 30)))
            autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))

            # 保存快照。
            if cur_tick % image_snapshot_ticks == 0 or done:
                grid_size, grid_reals, grid_labels, grid_latents = misc.setup_snapshot_image_grid(G, training_set, **grid_args)
                grid_fakes = Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=sched.minibatch // submit_config.num_gpus)
                grid_fakes0 = (grid_fakes * 0.5 + 0.5) * 255
                h5_data = grid_fakes0
                h5_data = h5_data.squeeze()
                with h5py.File('results/' + 'fake' + str(cur_tick) + '.hdf5', 'w') as f:
                    f.create_dataset('data', data=h5_data, dtype="i8", compression="gzip")


            if cur_tick % network_snapshot_ticks == 0 or done or cur_tick == 1:
                pkl = os.path.join(submit_config.run_dir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000))
                misc.save_pkl((G, D, Gs), pkl)

            # 更新摘要和RunContext。
            tflib.autosummary.save_summaries(summary_log, cur_nimg)
            ctx.update('%.2f' % sched.lod, cur_epoch=cur_nimg // 1000, max_epoch=total_kimg)
            maintenance_time = ctx.get_last_update_interval() - tick_time

    # 保存最终结果。
    misc.save_pkl((G, D, Gs), os.path.join(submit_config.run_dir, 'network-final.pkl'))
    summary_log.close()

    ctx.close()
