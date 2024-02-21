import os
import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras

import networks
import loss
import dataset

def saveModel(model, name, num):
    json = model.to_json()
    with open("model/"+name+".json", "w") as json_file:
        json_file.write(json)

    model.save_weights("model/"+name+"_"+str(num)+".h5")

def training_loop(
    dataset_dir,                        # Defect-free dataset folder (under ./data).
    defect_dir,                         # Defect dataset folder (under ./data).
    mask_dir,                           # Defect mask folder (under ./data).
    out_dir,                            # Output directory.
    img_res,                            # Image resolution
    sgan_kimg               = 3000,     # Total length of the training in stage 1
    dfmgan_kimg             = 40,       # Total length of the training in stage 2
    batch_size              = 1,        # Global minibatch size.
    G_ema_kimg              = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    G_ema_rampup            = None,     # EMA ramp-up coefficient.
    lazy_regularization     = True,     # Perform regularization as a separate training step?
    G_reg_interval          = 4,        # How often the perform regularization for G? Ignored if lazy_regularization=False.
    D_reg_interval          = 16,       # How often the perform regularization for D? Ignored if lazy_regularization=False.
    nimg_per_tick           = 400,      # Progress snapshot interval.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = only save 'networks-final.pkl'.
    learning_rate           = 0.0025,
    latent_size             = 512,

    # DFMGAN args
    ft = None,
    D_match_reg_interval = 16,
):
    # =============== Stage 1 : training backbone(StyleGAN2) ===============
    print("=====> Start backbone training...")

    # Load training set.
    training_set = dataset.load_dataset(dataset_dir, img_res)

    # Construct networks.
    G = networks.generator(img_resolution= img_res, batch_size= batch_size)
    D = networks.discriminator(activation= tf.nn.leaky_relu, img_resolution= img_res)
    # Gema = keras.models.clone_model(G)
    # Gema.set_weights(G.get_weights())

    # set up training optimizers
    phases = dict() 
    for name, reg_interval in [('G', G_reg_interval), ('D', D_reg_interval)]:
        if lazy_regularization:
            mb_ratio = reg_interval / (reg_interval + 1)
            lr = learning_rate * mb_ratio
            b1, b2 = [beta ** mb_ratio for beta in [0.0, 0.99]]
            opt = keras.optimizers.Adam(learning_rate= lr, beta_1= b1, beta_2= b2, epsilon= 1e-8)
            phases[name] = dict(opt= opt, interval= reg_interval)
        else: 
            phases[name] = dict(opt= opt, interval= 1)

    # start training
    print(f"=====> Training for {sgan_kimg} kimg...")
    curr_nimg = 0
    curr_tick = 0
    tick_start_nimg = curr_nimg
    batch_id = 0
    pl_penalty = loss.path_length_penalty(0.99)
    done = False
    while not done:
        images = dataset.get_batch(training_set, batch_size)
        gen_z = np.random.normal(0.0, 1.0, size= [batch_size, latent_size])
        
        do_g_reg = batch_id % phases['G']['interval'] == 0
        do_d_reg = batch_id % phases['D']['interval'] == 0

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            g_tape.watch(G.trainable_variables)
            d_tape.watch(D.trainable_variables)

            w1, w2, gen_imgs, mask = G([gen_z, None])

            fake_logits = D(gen_imgs)
            real_logits = D(images)

            g_loss = loss.g_logistic_loss(fake_logits)
            d_loss = loss.d_logistic_loss(fake_logits, real_logits)

            if do_g_reg:
                greg_loss = pl_penalty.call([w1, w2], G)
                g_loss += greg_loss
            if do_d_reg:
                dreg_loss = loss.r1_regularization(images, D)
                d_loss += dreg_loss
        
            g_grad = g_tape.gradient(g_loss, G.trainable_variables)
            d_grad = d_tape.gradient(d_loss, D.trainable_variables)

        phases['G']['opt'].apply_gradients(zip(g_grad, G.trainable_variables))
        phases['D']['opt'].apply_gradients(zip(d_grad, D.trainable_variables))

        curr_nimg += batch_size
        batch_id += 1
        done = (curr_nimg >= sgan_kimg * 1000)

        if (not done) and () and (curr_nimg < tick_start_nimg + nimg_per_tick):
            continue

        # Print Info
        print(f'Tick {curr_tick} / {sgan_kimg} : ')
        print(f'G : {g_loss}')
        print(f'D : {d_loss}')

        # Save model
        if (network_snapshot_ticks is not None) and (done or curr_tick % network_snapshot_ticks == 0):
            saveModel(G, "Gen", curr_tick // network_snapshot_ticks)
            saveModel(D, "Disc", curr_tick // network_snapshot_ticks)

        # Update
        curr_tick += 1
        tick_start_nimg = curr_nimg
        if done:
            break

    
    print("=====> Backbone training completed...")


    # =============== Stage 2 : training DFMGAN ===============
    print("=====> Start DFMGAN training...")

    # Load training set.
    defect_ds = dataset.load_dataset(defect_dir, img_res)
    mask_ds = dataset.load_dataset(mask_dir, img_res)
    
    # Construct networks.
    G.set_defect()
    # nGema.set_defect()
    Dmatch = networks.discriminator(activation= tf.nn.leaky_relu, img_resolution= img_res)

    # set up training optimizers 
    phases = dict() 
    training_nets = [('G', G_reg_interval), ('D', D_reg_interval), ('D_match', D_match_reg_interval)]
    for name, reg_interval in training_nets:
        if lazy_regularization:
            mb_ratio = reg_interval / (reg_interval + 1)
            lr = learning_rate * mb_ratio
            b1, b2 = [beta ** mb_ratio for beta in [0.0, 0.99]]
            opt = keras.optimizers.Adam(learning_rate= lr, beta_1= b1, beta_2= b2, epsilon= 1e-8)
            phases[name] = dict(opt= opt, interval= reg_interval)
        else: 
            phases[name] = dict(opt= opt, interval= 1)

    # start training
    print(f'=====> Training for {dfmgan_kimg} kimg...')
    curr_nimg = 0
    curr_tick = 0
    tick_start_nimg = curr_nimg
    batch_id = 0
    done = True
    while not done:
        gen_z = np.random.normal(0.0, 1.0, size= [len(phases), batch_size, latent_size])
        defect_z = np.random.normal(0.0, 1.0, size= [len(phases), batch_size, latent_size])
        
        do_g_reg = batch_id % phases['G']['interval'] == 0
        do_d_reg = batch_id % phases['D']['interval'] == 0
        do_dm_reg = batch_id % phases['D_match']['interval'] == 0

        curr_nimg += batch_size
        batch_id += 1
        done = (curr_nimg >= sgan_kimg * 1000)

        if (not done) and () and (curr_nimg < tick_start_nimg + nimg_per_tick):
            continue

        # Print Info
        print(f'Tick {curr_tick} / {sgan_kimg} : ')
        print(f'G       : {g_loss}')
        print(f'D       : {d_loss}')
        print(f'D_match : {dm_loss}')

        # Save model
        if (network_snapshot_ticks is not None) and (done or curr_tick % network_snapshot_ticks == 0):
            saveModel(G, "Gen", curr_tick // network_snapshot_ticks)
            saveModel(D, "Disc", curr_tick // network_snapshot_ticks)
            saveModel(Dmatch, "Disc_match", curr_tick // network_snapshot_ticks)

        # Update
        curr_tick += 1
        tick_start_nimg = curr_nimg
        if done:
            break

    print("=====> Exiting...")



training_loop("./data/hazelnut/train/good/", "./data/hazelnut/test/hole/", "./data/hazelnut/ground_truth/hole/", "./run/", 256)