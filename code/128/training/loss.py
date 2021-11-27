# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.


import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

#----------------------------------------------------------------------------
# 便捷函数，可将其所有参数转换为tf.float32。

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# WGAN和WGAN-GP损失函数。

def G_wgan(G, D, opt, training_set, minibatch_size): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -fake_scores_out
    return loss  # Loss_G = -D(G(z))

def D_wgan(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_epsilon = 0.001):  # ε项的值， \epsilon_{drift}.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss  # Loss_D = D(G(z)) - D(x) + ε·D(x)^2

def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_lambda     = 10.0,     # 梯度惩罚项的权重。
    wgan_epsilon    = 0.001,    # ε项的值, \epsilon_{drift}.
    wgan_target     = 1.0):     # 梯度幅度的目标值，即满足1-lipschitz范式所以梯度的模得小于等于1。

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)  # alpha
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
            # 惩罚区的样本为xp = alpha * x + (1 - alpha) * G(z)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))  # 惩罚区样本的判别值D(xp)
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))  # 惩罚区样本的梯度∇T
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3,4]))  # 惩罚区样本梯度∇T的模||∇T||
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)  # 惩罚项为(||∇T||-1)^2
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss  # Loss_D = D(G(z)) - D(x) + η·(||∇T||-1)^2 + ε·D(x)^2

#----------------------------------------------------------------------------
# 铰链损失函数(Hinge Loss)（这些与WGAN中的G损失一起使用）

def D_hinge(G, D, opt, training_set, minibatch_size, reals, labels): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.maximum(0., 1.+fake_scores_out) + tf.maximum(0., 1.-real_scores_out)
    return loss  # Loss_D = max(0,1+D(G(z))) + max(0,1-D(x))

def D_hinge_gp(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_lambda     = 10.0,     # 梯度惩罚项的权重。
    wgan_target     = 1.0):     # 梯度幅度的目标值。

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.maximum(0., 1.+fake_scores_out) + tf.maximum(0., 1.-real_scores_out)

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3,4]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    return loss  # Loss_D = max(0,1+D(G(z))) + max(0,1-D(x)) + η·(||∇T||-1)^2


#----------------------------------------------------------------------------

def G_logistic_saturating(G, D, opt, training_set, minibatch_size):  # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -tf.nn.softplus(fake_scores_out)
    return loss  # Loss_G = -log(exp(D(G(z))) + 1)

def G_logistic_nonsaturating(G, D, opt, training_set, minibatch_size): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = tf.nn.softplus(-fake_scores_out)
    return loss  # Loss_G = log(exp(-D(G(z))) + 1)     #用的

def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out)
    loss += tf.nn.softplus(-real_scores_out)
    return loss  # Loss_D = log(exp(D(G(z))) + 1) + log(exp(-D(x)) + 1)

def D_logistic_simplegp(G, D, opt, training_set, minibatch_size, reals, labels, r1_gamma=10.0, r2_gamma=0.0): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out)
    loss += tf.nn.softplus(-real_scores_out)
    if r1_gamma != 0.0:
        with tf.name_scope('R1Penalty'):
            real_loss = opt.apply_loss_scaling(tf.reduce_sum(real_scores_out))  # 惩罚区(来自真实样本)的判别值D(x_real)
            real_grads = opt.undo_loss_scaling(fp32(tf.gradients(real_loss, [reals])[0]))  # 惩罚区样本的梯度∇T_real
            r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3,4])  # 惩罚项为∑(∇T_real^2)
            r1_penalty = autosummary('Loss/r1_penalty', r1_penalty)
        loss += r1_penalty * (r1_gamma * 0.5)

    if r2_gamma != 0.0:
        with tf.name_scope('R2Penalty'):
            fake_loss = opt.apply_loss_scaling(tf.reduce_sum(fake_scores_out))  # 惩罚区(来自生成样本)的判别值D(x_fake)
            fake_grads = opt.undo_loss_scaling(fp32(tf.gradients(fake_loss, [fake_images_out])[0]))  # 惩罚区样本的梯度∇T_fake
            r2_penalty = tf.reduce_sum(tf.square(fake_grads), axis=[1,2,3,4])  # 惩罚项为∑(∇T_fake^2)
            r2_penalty = autosummary('Loss/r2_penalty', r2_penalty)
        loss += r2_penalty * (r2_gamma * 0.5)
    return loss  # Loss_D = log(exp(D(G(z))) + 1) + log(exp(-D(x)) + 1) + r1_gamma*0.5*∑(∇T_real^2) + r2_gamma*0.5*∑(∇T_fake^2)

#----------------------------------------------------------------------------
