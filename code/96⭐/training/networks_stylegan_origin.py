# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.

#----------------------------------------------------------------------------

def _blur2d(x, f=[1,2,1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 5 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis, np.newaxis] * f[np.newaxis, :, np.newaxis] * f[np.newaxis, np.newaxis, :]
    assert f.ndim == 3

    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1, ::-1]
    f = f[:, :, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, 1, int(x.shape[1]), int(x.shape[1])])

    if f.shape == (1, 1) and f[0, 0] == 1:
        return x

    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, 1, stride, stride, stride]
    x = tf.nn.conv3d(x, f, strides=strides, padding='SAME', data_format='NCDHW') #no tf.nn.depthwise_conv3d
    x = tf.cast(x, orig_dtype)
    return x

def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 5 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    if gain != 1:
        x *= gain

    if factor == 1:
        return x

    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1, s[4], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor, s[4] * factor])
    return x

def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 5 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    if gain != 1:
        x *= gain

    if factor == 1:
        return x

    ksize = [1, 1, factor, factor, factor] #已查
    return tf.nn.avg_pool3d(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCDHW')

#----------------------------------------------------------------------------

def blur2d(x, f=[1,2,1], normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)
            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)
            return y, grad
        return func(x)

def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor**2)
                return dx, lambda ddx: _upscale2d(ddx, factor)
            return y, grad
        return func(x)

def downscale2d(x, factor=2):
    with tf.variable_scope('Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1/factor**2)
                return dx, lambda ddx: _downscale2d(ddx, factor)
            return y, grad
        return func(x)

#----------------------------------------------------------------------------

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1):
    fan_in = np.prod(shape[:-1])
    he_std = gain / np.sqrt(fan_in)  # 0.5*n*var(w)=1 , so：std(w)=sqrt(2)/sqrt(n)=gain/sqrt(fan_in)

    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable('weight', shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------

def dense(x, fmaps, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------

def conv2d(x, fmaps, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv3d(x, w, strides=[1,1,1,1,1], padding='SAME', data_format='NCDHW')

#----------------------------------------------------------------------------

def upscale2d_conv2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) * 2 >= 16  # x的高和宽≥8的话使用融合，否则不使用融合

    if not fused_scale:
        return conv2d(upscale2d(x), fmaps, kernel, **kwargs)

    w = get_weight([kernel, kernel, kernel, x.shape[1].value, fmaps], **kwargs)  # 创建一个卷积层w，shape为[kernel, kernel, kernel, fmaps_in, fmaps_out]
    w = tf.transpose(w, [0, 1, 2, 4, 3])  # 将w转成[kernel, kernel, kernel, fmaps_out, fmaps_in]
    print('n_s 200 w.shape=', w.shape) #(3, 3, 3, 48, 96)
    w = tf.pad(w, [[1,1], [1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')  # 对w进行填充，在三个kernel的第0维和最后一维分别填充0，w的shape变为[kernel+2, kernel+2, kernel+2, fmaps_in, fmaps_out]
    print('n_s 202 w.shape=', w.shape) #(5, 5, 5, 48, 96)
    w = tf.add_n([w[1:, 1:, 1:], w[1:, 1:, :-1], w[1:, :-1, 1:],w[1:, :-1, :-1], w[:-1, 1:, :-1], w[:-1, 1:, 1:], w[:-1, :-1, 1:], w[:-1, :-1, :-1]])  # 填充区域求和两次，非填充区域求和八次
    print('n_s 204 w.shape,w[1:, 1:, 1:].shape=', w.shape, w[1:, 1:, 1:].shape) #(4, 4, 4, 48, 96) (3, 3, 3, 48, 96)
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2]
    return tf.nn.conv3d_transpose(x, w, os, strides=[1,1,2,2,2], padding='SAME', data_format='NCDHW')

def conv2d_downscale2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) * 2 >= 16  # x的高和宽≥8的话使用融合，否则不使用融合

    if not fused_scale:
        return downscale2d(conv2d(x, fmaps, kernel, **kwargs))

    w = get_weight([kernel, kernel, kernel,x.shape[1].value, fmaps], **kwargs)  # 创建一个卷积层w，shape为[kernel, kernel, kernel, fmaps_in, fmaps_out]
    w = tf.pad(w, [[1,1] ,[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')  # 对w进行填充，在三个kernel的第0维和最后一维分别填充0，w的shape变为[kernel+2, kernel+2, kernel+2, fmaps_in, fmaps_out]
    w = tf.add_n([w[1:, 1:, 1:], w[1:, 1:, :-1], w[1:, :-1, 1:],w[1:, :-1, :-1], w[:-1, 1:, :-1], w[:-1, 1:, 1:], w[:-1, :-1, 1:], w[:-1, :-1, :-1]]) * 0.125  # 填充区域求和两次，非填充区域求和八次
    w = tf.cast(w, x.dtype)
    return tf.nn.conv3d(x, w, strides=[1,1,2,2,2], padding='SAME', data_format='NCDHW')

#----------------------------------------------------------------------------

def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1, 1])

#----------------------------------------------------------------------------

def leaky_relu(x, alpha=0.2):
    with tf.variable_scope('LeakyReLU'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        @tf.custom_gradient
        def func(x):
            y = tf.maximum(x, x * alpha)
            @tf.custom_gradient
            def grad(dy):
                dx = tf.where(y >= 0, dy, dy * alpha)
                return dx, lambda ddx: tf.where(y >= 0, ddx, ddx * alpha)
            return y, grad
        return func(x)

#----------------------------------------------------------------------------

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)  # x = x/√(x2_avg+ε)

#----------------------------------------------------------------------------

def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 5  # NCDHW
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[2,3,4], keepdims=True)  # 实例归一化仅对DHW做归一化，所以axis是2,3,4
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2,3,4], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)  # x = (x-x_mean)/√(x2_avg+ε)
        return x

#----------------------------------------------------------------------------
# 样式调制（AdaIN）

def style_mod(x, dlatent, **kwargs):
    with tf.variable_scope('StyleMod'):
        style = apply_bias(dense(dlatent, fmaps=x.shape[1]*2, gain=1, **kwargs))  # 仿射变换A（通过全连接层将dlatent扩大一倍）
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))  # 扩大后的dlatent转换为放缩因子y_s,i和偏置因子y_b,i
        return x * (style[:,0] + 1) + style[:,1]  # 对卷积后的x执行样式调制（自适应实例归一化）

#----------------------------------------------------------------------------
# 噪音输入。

def apply_noise(x, noise_var=None, randomize_noise=True):
    assert len(x.shape) == 5  # NCDHW
    with tf.variable_scope('Noise'):
        if noise_var is None or randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3], x.shape[4]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_var, x.dtype)
        weight = tf.get_variable('weight', shape=[x.shape[1].value], initializer=tf.initializers.zeros())  # 噪音的权重
        return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1, 1])

#----------------------------------------------------------------------------
# 小批量标准偏差。

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])     # 小批量一定能被group_size整除（或小于group_size）。
        s = x.shape                                             # [NC DHW]  输入shape.
        y = tf.reshape(x, [group_size, -1, num_new_features, s[1]//num_new_features, s[2], s[3], s[4]])   # [4n1C DHW] 将小批量拆分为M个大小为G的组。将通道拆分为n个c个通道的组。
        y = tf.cast(y, tf.float32)                              # [4n1C DHW] 转换成FP32。
        y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [4n1C DHW] 按组减去均值。
        y = tf.reduce_mean(tf.square(y), axis=0)                # [n1C DHW]  按组计算方差。
        y = tf.sqrt(y + 1e-8)                                   # [n1C DHW]  按组计算标准方差。
        y = tf.reduce_mean(y, axis=[2,3,4,5], keepdims=True)      # [n11 111]  在特征图和像素上采取平均值。
        y = tf.reduce_mean(y, axis=[2])                         # [n1 111] 将通道划分为c个通道组。
        y = tf.cast(y, x.dtype)                                 # [n1 111]  转换回原始的数据类型。
        y = tf.tile(y, [group_size, 1, s[2], s[3], s[4]])             # [N1 DHW]  按组和像素进行复制。
        return tf.concat([x, y], axis=1)                        # [N(C+1) DHW]  添加为新的特征图。

#----------------------------------------------------------------------------

def G_style(
    latents_in,                                     # 第一个输入：Z码向量 [minibatch, latent_size].
    labels_in,                                      # 第二个输入：条件标签 [minibatch, label_size].
    truncation_psi          = 0.7,                  # 截断技巧的样式强度乘数。 None = disable.
    truncation_cutoff       = 8,                    # 要应用截断技巧的层数。 None = disable.
    truncation_psi_val      = None,                 # 验证期间要使用的truncation_psi的值。
    truncation_cutoff_val   = None,                 # 验证期间要使用的truncation_cutoff的值。
    dlatent_avg_beta        = 0.995,                # 在训练期间跟踪W的移动平均值的衰减率。 None = disable.
    style_mixing_prob       = 0.9,                  # 训练期间混合样式的概率。 None = disable.
    is_training             = False,                # 网络正在接受训练？ 这个选择可以启用和禁用特定特征。
    is_validation           = False,                # 网络正在验证中？ 这个选择用于确定truncation_psi的值。
    is_template_graph       = False,                # True表示由Network类构造的模板图，False表示实际评估。
    components              = dnnlib.EasyDict(),    # 子网络的容器。调用时候保留。
    **kwargs):                                      # 子网络的参数们 (G_mapping 和 G_synthesis)。

    # 参数验证。
    assert not is_training or not is_validation
    assert isinstance(components, dnnlib.EasyDict)
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training or (truncation_cutoff is not None and not tflib.is_tf_expression(truncation_cutoff) and truncation_cutoff <= 0):
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing_prob is not None and not tflib.is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None

    if 'synthesis' not in components:
        components.synthesis = tflib.Network('G_synthesis', func_name=G_synthesis, **kwargs)
    num_layers = components.synthesis.input_shape[1]
    dlatent_size = components.synthesis.input_shape[2]
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=G_mapping, dlatent_broadcast=num_layers, **kwargs)

    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)  # 初始化为0。lod的定义式为：lod = resolution_log2 - res
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)

    dlatents = components.mapping.get_output_for(latents_in, labels_in, **kwargs)

    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    if style_mixing_prob is not None:
        with tf.name_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, **kwargs)
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

    if truncation_psi is not None and truncation_cutoff is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_idx.shape, dtype=np.float32)
            coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            dlatents = tflib.lerp(dlatent_avg, dlatents, coefs)

    with tf.control_dependencies([tf.assign(components.synthesis.find_var('lod'), lod_in)]):
        images_out = components.synthesis.get_output_for(dlatents, force_clean_graph=is_template_graph, **kwargs)
    return tf.identity(images_out, name='images_out')

#----------------------------------------------------------------------------

def G_mapping(
    latents_in,                             # 第一个输入：Z码向量 [minibatch, latent_size].
    labels_in,                              # 第二个输入：条件标签 [minibatch, label_size].
    latent_size             = 128,          # 潜在向量（Z）维度。
    label_size              = 0,            # 标签尺寸，0表示没有标签。
    dlatent_size            = 128,          # 解缠后的中间向量 (W) 维度。
    dlatent_broadcast       = None,         # 将解缠后的中间向量（W）输出为[minibatch，dlatent_size]或[minibatch，dlatent_broadcast，dlatent_size]格式。
    mapping_layers          = 8,            # 映射网络的层数。
    mapping_fmaps           = 128,          # 映射层中的特征图维度。
    mapping_lrmul           = 0.01,         # 映射层的学习率变化率。
    mapping_nonlinearity    = 'lrelu',      # 激活函数: 'relu', 'lrelu'.
    use_wscale              = True,         # 启用均等的学习率？
    normalize_latents       = True,         # 在将潜在向量（Z）馈送到映射层之前对其进行归一化？
    dtype                   = 'float32',    # 用于激活和输出的数据类型。
    **_kwargs):                             # 忽略无法识别的关键字参数。

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[mapping_nonlinearity]

    # 输入
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # 嵌入标签并将其与潜码连接起来。
    if label_size:
        with tf.variable_scope('LabelConcat'):
            w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # 归一化潜码。
    if normalize_latents:
        x = pixel_norm(x)

    # 映射层。
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = dense(x, fmaps=fmaps, gain=gain, use_wscale=use_wscale, lrmul=mapping_lrmul)
            x = apply_bias(x, lrmul=mapping_lrmul)
            x = act(x)

    # 广播。
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):   # 简单的复制扩充
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # 输出。
    assert x.dtype == tf.as_dtype(dtype)
    return tf.identity(x, name='dlatents_out')

#----------------------------------------------------------------------------

def G_synthesis(
    dlatents_in,                        # 输入：解缠的中间向量 (W) [minibatch, num_layers, dlatent_size].
    dlatent_size        = 128,          # 解缠的中间向量 (W) 的维度。
    num_channels        = 1,            # 输出颜色通道数。
    resolution          = 64,           # 输出分辨率。
    fmap_base           = 384,          # 特征图的总数目
    fmap_decay          = 1.0,          # 当分辨率翻倍时以log2降低特征图，这儿指示降低的速率。
    fmap_max            = 192,          # 在任何层中特征图的最大数量。
    use_styles          = True,         # 启用样式输入？
    const_input_layer   = True,         # 第一层是常数？
    use_noise           = True,         # 启用噪音输入？
    randomize_noise     = True,         # True表示每次都随机化噪声输入（不确定），False表示从变量中读取噪声输入。
    close_deep_noise    = False,        # 关闭深层噪声？
    close_deep_noise_x  = 6,            # 从哪层开始关闭？
    close_deep_style    = False,        # 关闭深层噪声？
    close_deep_style_x  = 6,           # 从哪层开始关闭？
    nonlinearity        = 'lrelu',      # 激活函数: 'relu', 'lrelu'
    use_wscale          = True,         # 启用均等的学习率？
    use_pixel_norm      = False,        # 启用逐像素特征向量归一化？
    use_instance_norm   = True,         # 启用实例规一化？
    dtype               = 'float32',    # 用于激活和输出的数据类型。
    fused_scale         = 'auto',       # True = 融合卷积+缩放，False = 单独操作，'auto'= 自动决定。
    blur_filter         = [1,2,1],      # 重采样激活时应用的低通卷积核（Low-pass filter）。None表示不过滤。
    structure           = 'auto',       # 'fixed' = 无渐进式增长，'linear' = 人类可读，'recursive' = 有效，'auto' = 自动选择。
    is_template_graph   = False,        # True表示由Network类构造的模板图，False表示实际评估。
    force_clean_graph   = False,        # True表示构建一个在TensorBoard中看起来很漂亮的干净图形，False表示默认设置。
    **_kwargs):                         # 忽略无法识别的关键字参数。

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def blur(x): return blur2d(x, blur_filter) if blur_filter else x
    if is_template_graph: force_clean_graph = True
    if force_clean_graph: randomize_noise = False
    if structure == 'auto': structure = 'linear' if force_clean_graph else 'recursive'
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearity]
    num_layers = resolution_log2 * 2 - 2
    num_styles = num_layers if use_styles else 1
    images_out = None

    # 主要输入。
    dlatents_in.set_shape([None, num_styles, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)

    # 创建噪音。
    noise_inputs = []
    if use_noise:
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, use_noise, 6*2**(res-2), 6*2**(res-2), 6*2**(res-2)]
            noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    def layer_epilogue(x, layer_idx):
        if use_noise:
            if close_deep_noise:
                if layer_idx < close_deep_noise_x:
                    x = apply_noise(x, noise_inputs[layer_idx], randomize_noise=randomize_noise)
            else:
                x = apply_noise(x, noise_inputs[layer_idx], randomize_noise=randomize_noise)
        x = apply_bias(x)
        x = act(x)
        if use_pixel_norm:
            x = pixel_norm(x)
        if use_instance_norm:
            x = instance_norm(x)
        if use_styles:
            if close_deep_style:
                if layer_idx < close_deep_style_x:
                    x = style_mod(x, dlatents_in[:, layer_idx], use_wscale=use_wscale)
            else:
                x = style_mod(x, dlatents_in[:, layer_idx], use_wscale=use_wscale)
        return x

    with tf.variable_scope('6x6x6'):
        if const_input_layer:
            with tf.variable_scope('Const'):
                x = tf.get_variable('const', shape=[1, nf(1), 6, 6, 6], initializer=tf.initializers.ones())
                x = layer_epilogue(tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1, 1]), 0)
        else:
            with tf.variable_scope('Dense'):
                x = dense(dlatents_in[:, 0], fmaps=nf(1)*16, gain=gain/4, use_wscale=use_wscale)
                x = layer_epilogue(tf.reshape(x, [-1, nf(1), 6, 6, 6]), 0)
        with tf.variable_scope('Conv'):
            x = layer_epilogue(conv2d(x, fmaps=nf(1), kernel=3, gain=gain, use_wscale=use_wscale), 1)

    def block(res, x):
        with tf.variable_scope('%dx%dx%d' % (6*2**(res-2), 6*2**(res-2), 6*2**(res-2))):
            with tf.variable_scope('Conv0_up'):
                x = layer_epilogue(blur(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)), res*2-4)
            with tf.variable_scope('Conv1'):
                x = layer_epilogue(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale), res*2-3)
            return x
        
    def togray(res, x):
        lod = resolution_log2 - res
        with tf.variable_scope('ToGRAY_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=use_wscale))
        

    # 固定结构
    if structure == 'fixed':
        for res in range(3, resolution_log2 + 1):
            x = block(res, x)
        images_out = togray(resolution_log2, x)
        images_out = tf.nn.tanh(images_out)  # 不参与循环

    # 线性结构
    if structure == 'linear':
        print('linear')
        images_out = togray(2, x)
        for res in range(3, resolution_log2 + 1):
            lod = resolution_log2 - res
            x = block(res, x)
            img = togray(res, x)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = tflib.lerp_clip(img, images_out, lod_in - lod)
        with tf.variable_scope('Tanh'):
            images_out = tf.nn.tanh(images_out)  # 不参与循环

    # 递归结构：默认。
    if structure == 'recursive':
        print('recursive')
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)  # 返回一个函数，依据是否满足new_cond决定返回new_lambda函数还是cur_lambda函数
        def grow(x, res, lod):
            y = block(res, x)
            img = lambda: upscale2d(togray(res, y), 2**lod)
            img = cset(img, (lod_in > lod), lambda: upscale2d(tflib.lerp(togray(res, y), upscale2d(togray(res - 1, x)), lod_in - lod), 2**lod))  # 如果输入层数lod_in超过当前层lod的话（但同时小于lod+1），实现从lod对应分辨率到lod_in对应分辨率的扩增，采用线性插值；否则按lod处理。
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))  # 如果lod_in小于lod且不是最后一层的话（也就是前者的res超过后者的res），表明可以进入到下一级分辨率上了，此时res+1, lod-1
            return img()
        images_out = grow(x, 3, resolution_log2 - 3)  # res一开始为3，lod一开始为resolution_log2 - res，利用递归就可以构建res从3增加到resolution_log2的全部架构
        images_out = tf.nn.tanh(images_out)  # 不参与循环

    assert images_out.dtype == tf.as_dtype(dtype)
    print('572 success')
    return tf.identity(images_out, name='images_out')


#----------------------------------------------------------------------------

def D_basic(
    images_in,                          # 第一个输入：图片 [minibatch, channel, height, width].
    labels_in,                          # 第二个输入：标签 [minibatch, label_size].
    num_channels        = 1,            # 输入颜色通道数。 根据数据集覆盖。
    resolution          = 64,           # 输入分辨率。 根据数据集覆盖。
    label_size          = 0,            # 标签的维数，0表示没有标签。根据数据集覆盖。
    fmap_base           = 384,          # 特征图的总数目
    fmap_decay          = 1.0,          # 当分辨率翻倍时以log2降低特征图，这儿指示降低的速率。
    fmap_max            = 192,          # 在任何层中特征图的最大数量。
    nonlinearity        = 'lrelu',      # 激活函数: 'relu', 'lrelu'。
    use_wscale          = True,         # 启用均等的学习率？
    mbstd_group_size    = 4,            # 小批量标准偏差层的组大小，0表示禁用。
    mbstd_num_features  = 1,            # 小批量标准偏差层的特征数量。
    dtype               = 'float32',    # 用于激活和输出的数据类型。
    fused_scale         = 'auto',       # True = 融合卷积+缩放，False = 单独操作，'auto'= 自动决定。
    blur_filter         = [1,2,1],      # 重采样激活时应用的低通卷积核（Low-pass filter）。None表示不过滤。
    structure           = 'auto',       # 'fixed' = 无渐进式增长，'linear' = 人类可读，'recursive' = 有效，'auto' = 自动选择。
    is_template_graph   = False,        # True表示由Network类构造的模板图，False表示实际评估。
    **_kwargs):                         # 忽略无法识别的关键字参数。

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    def blur(x): return blur2d(x, blur_filter) if blur_filter else x
    if structure == 'auto': structure = 'linear' if is_template_graph else 'recursive'
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearity]
    # 输入处理
    images_in.set_shape([None, num_channels, resolution/4*6, resolution/4*6, resolution/4*6])
    labels_in.set_shape([None, label_size])
    images_in = tf.cast(images_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)
    scores_out = None

    # 构建block块。
    def fromgray(x, res):
        with tf.variable_scope('FromGRAY_lod%d' % (resolution_log2 - res)):
            return act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, gain=gain, use_wscale=use_wscale)))
    def block(x, res):
        with tf.variable_scope('%dx%dx%d' % (6*2**(res-2), 6*2**(res-2), 6*2**(res-2))):
            if res >= 3:  # 8x8x8分辨率及以上
                with tf.variable_scope('Conv0'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Conv1_down'):
                    x = act(apply_bias(conv2d_downscale2d(blur(x), fmaps=nf(res-2), kernel=3, gain=gain, use_wscale=use_wscale, fused_scale=fused_scale)))
            else:  # 4x4x4分辨率，得到判别分数scores_out
                if mbstd_group_size > 1:
                    x = minibatch_stddev_layer(x, mbstd_group_size, mbstd_num_features)
                with tf.variable_scope('Conv'):
                    x = act(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense0'):
                    x = act(apply_bias(dense(x, fmaps=nf(res-2), gain=gain, use_wscale=use_wscale)))
                with tf.variable_scope('Dense1'):
                    x = apply_bias(dense(x, fmaps=max(label_size, 1), gain=1, use_wscale=use_wscale))
            return x

    # 固定结构。
    if structure == 'fixed':
        x = fromgray(images_in, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            x = block(x, res)
        scores_out = block(x, 2)

    # 线性结构。
    if structure == 'linear':
        img = images_in
        x = fromgray(img, resolution_log2)
        for res in range(resolution_log2, 2, -1):
            lod = resolution_log2 - res
            x = block(x, res)
            img = downscale2d(img)
            y = fromgray(img, res - 1)
            with tf.variable_scope('Grow_lod%d' % lod):
                x = tflib.lerp_clip(x, y, lod_in - lod)
        scores_out = block(x, 2)

    # 递归结构：默认。
    if structure == 'recursive':  # 注意判别器在训练时是输入图片先进入lod最小的层，但是构建判别网络时是lod从大往小构建，所以递归的过程是与生成器相反的。
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)  # 返回一个函数，依据是否满足new_cond决定返回new_lambda函数还是cur_lambda函数
        def grow(res, lod):
            x = lambda: fromgray(downscale2d(images_in, 2**lod), res)  # 先暂时将下采样函数赋给x
            if lod > 0: x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))  # 非第一层时，如果输入层数lod_in小于当前层lod的话，表明可以进入到下一级分辨率上了，将grow()赋给x；否则x还是保留为下采样函数。
            x = block(x(), res); y = lambda: x  # x执行一次自身的函数，构建出一个block，并将结果赋给y（以函数的形式）
            if res > 2: y = cset(y, (lod_in > lod), lambda: tflib.lerp(x, fromgray(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))  # 非最后一层时，如果输入层数lod_in大于当前层lod的话，表明需要进行插值操作，将lerp()赋给y；否则y还是保留为之前的操作。
            return y()
        scores_out = grow(2, resolution_log2 - 2)  # 构建判别网络时是lod从大往小构建

    if label_size:
        with tf.variable_scope('LabelSwitch'):
            scores_out = tf.reduce_sum(scores_out * labels_in, axis=1, keepdims=True)

    assert scores_out.dtype == tf.as_dtype(dtype)
    scores_out = tf.identity(scores_out, name='scores_out')
    return scores_out  # 输出

#----------------------------------------------------------------------------
