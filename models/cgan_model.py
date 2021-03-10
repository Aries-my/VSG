from __future__ import print_function, division

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam, SGD
from network import build_discriminator, build_generator

from plotting import plot_scatter_density
from IPython import display
import matplotlib.pyplot as plt
from dataset import get_true_x_give_y
from scipy import stats as st

"""
CGAN code derived from https://github.com/eriklindernoren/Keras-GAN
Generator will input (x & noise) and will output Ypred.
Discriminator will input (x & Ypred) and will differentiate between that and Y.
"""

def make_distance_plot(dist_list, label):
    plt.plot(range(0, (len(dist_list)) * 10, 10), dist_list, markersize=1, label=label)
    plt.title('Distance')
    plt.xlabel('Updates of the generator')
    plt.legend(loc="upper right")
    # plt.ylim(0, 0.5)
    plt.show()

class CGAN():
    def __init__(self, exp_config):
        if exp_config.model.optim_gen == "Adam":
            self.optimizer_gen = Adam(exp_config.model.lr_gen, decay=exp_config.model.dec_gen)
        else:
            self.optimizer_gen = SGD(exp_config.model.lr_gen, decay=exp_config.model.dec_gen)
        if exp_config.model.optim_disc == "Adam":
            self.optimizer_disc = Adam(exp_config.model.lr_disc, decay=exp_config.model.dec_disc)
        else:
            self.optimizer_disc = SGD(exp_config.model.lr_disc, decay=exp_config.model.dec_disc)
        self.activation = exp_config.model.activation
        self.seed = exp_config.model.random_seed
        self.scenario = exp_config.dataset.scenario

        if self.scenario == "CA-housing":
            self.x_input_size = 8
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 3
        elif self.scenario == "ailerons":
            self.x_input_size = 40
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 3
        elif self.scenario == "comp-activ":
            self.x_input_size = 21
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 5
        elif self.scenario == "pumadyn":
            self.x_input_size = 32
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 4
        elif self.scenario == "bank":
            self.x_input_size = 32
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 4
            self.architecture = 3
        elif self.scenario == "census-house":
            self.x_input_size = 16
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 3
        elif self.scenario == "abalone":
            self.x_input_size = 7
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 3
        elif self.scenario == "hdpe":
            self.x_input_size = 15
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 6
        elif self.scenario == "magical_sinus":
            self.x_input_size = 2
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 7
        elif self.scenario == "MLCC":
            self.x_input_size = 12
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            self.architecture = 7
        else:
            self.x_input_size = 1
            self.y_input_size = 1
            self.z_input_size = exp_config.model.z_input_size
            if self.scenario == "linear" or self.scenario == "sinus":
                self.architecture = 1
            else:
                self.architecture = 2

        if exp_config.model.architecture is not None:
            self.architecture = exp_config.model.architecture

        # Build and compile the discriminator
        self.discriminator = build_discriminator(self)
        self.discriminator.compile(
            loss=['binary_crossentropy'],
            optimizer=self.optimizer_disc,
            metrics=['accuracy'])

        # Build the generator
        self.generator = build_generator(self)

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.z_input_size,))

        y = Input(shape=(self.y_input_size,))
        x_gen = self.generator([noise, y])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        validity = self.discriminator([x_gen, y])

        # The combined model (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, y], validity)
        self.combined.compile(
            loss=['binary_crossentropy'],
            optimizer=self.optimizer_gen)

        # Print network's architecture
        print(self.generator.summary())
        print(self.discriminator.summary())

    def train(self, xtrain, ytrain, epochs, batch_size=128, verbose=True):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        dLossErr = np.zeros([epochs, 1])
        dLossReal = np.zeros([epochs, 1])
        dLossFake = np.zeros([epochs, 1])
        gLossErr = np.zeros([epochs, 1])
        
        self.train_hist = {}
        self.train_hist['KL1_dist'] = []
        self.train_hist['KL2_dist'] = []
        self.train_hist['JS_dist'] = []

        for epoch in range(epochs):

            # train Generator and Discriminator alternatively.
            for iter in range(int(xtrain.shape[0] // batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Select a random half batch of samples
                idx = np.random.randint(0, xtrain.shape[0], batch_size)
                x, y = xtrain[idx], ytrain[idx]
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.z_input_size))
                # Generate a half batch of new images
                x_gen = self.generator.predict([noise, y])
                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([x, y], valid)
                d_loss_fake = self.discriminator.train_on_batch([x_gen, y], fake)
                d_loss = np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------
                # Condition on x

                # idx = np.random.randint(0, xtrain.shape[0], batch_size)
                # sample = xtrain[idx]

                # Train the generator
                g_loss = self.combined.train_on_batch([noise, y], valid)

            dLossErr[epoch] = d_loss[0]
            dLossReal[epoch] = d_loss_real[0]
            dLossFake[epoch] = d_loss_fake[0]
            gLossErr[epoch] = g_loss

            if verbose:
                print(f"Epoch: {epoch} / dLoss: {d_loss[0]} / gLoss: {g_loss}")

            # display probability density estimation for where inputs are 2-d cases.
            if self.x_input_size == 2:
                def _display_density(given_y, num_sampling=10000):
                    true_x = get_true_x_give_y(given_y)
                    true_y = given_y * np.ones((num_sampling, 1))
                    gen_x = self.predict(true_y)

                    plot_scatter_density(true_x[:, 0], true_x[:, 1],
                                         title=f"True $p(x|y={given_y})$ at epoch:{epoch + 1}")
                    plot_scatter_density(gen_x[:, 0], gen_x[:, 1],
                                         title=f"CGAN $p(x|y={given_y})$ at epoch:{epoch + 1}")

                if (epoch + 1) % 5000 == 0:
                    display.clear_output(wait=True)
                    _display_density(given_y=2.098)
                    KL1_sum = 0
                    KL2_sum = 0
                    JS_sum = 0
                    for iy in [2.098]:
                        true_x = get_true_x_give_y(iy)
                        true_y = iy * np.ones((10000, 1))
                        gen_x = self.predict(true_y)
                                    
                        nBins = 1000
                        MyRange = [0, 1]
                        true_cond_dist, _, _ = plt.hist(true_x, bins=nBins, range=MyRange, density=True, histtype='step')
                        gen_cond_dist, _, _ = plt.hist(x_gen, bins=nBins, range=MyRange, density=True, histtype='step')
                        plt.close()
                        true_cond_dist = true_cond_dist.T
                        gen_cond_dist = gen_cond_dist.T
                        
                        true_cond_dist1 = true_cond_dist[:, 0]
                        true_cond_dist2 = true_cond_dist[:, 1]
                        gen_cond_dist1 = gen_cond_dist[:, 0]
                        gen_cond_dist2 = gen_cond_dist[:, 1]
                        
                        true_n_gen_cond1 = np.concatenate((np.expand_dims(true_cond_dist1, axis=1), np.expand_dims(gen_cond_dist1, axis=1)),axis=1)
                        true_n_gen_cond1 = true_n_gen_cond1[true_n_gen_cond1[:, 1] != 0, :]  # 让第二列不为0
                        
                        true_n_gen_cond2 = np.concatenate((np.expand_dims(true_cond_dist2, axis=1), np.expand_dims(gen_cond_dist2, axis=1)),axis=1)
                        true_n_gen_cond2 = true_n_gen_cond2[true_n_gen_cond2[:, 1] != 0, :]  # 让第二列不为0
                        
                        KL1 = st.entropy(true_n_gen_cond1[:, 0], true_n_gen_cond1[:, 1])
                        KL2 = st.entropy(true_n_gen_cond2[:, 0], true_n_gen_cond2[:, 1])
                        m = 0.5 * (true_cond_dist + gen_cond_dist)
                        JS = 0.5 * (st.entropy(true_cond_dist, m) + st.entropy(gen_cond_dist, m))
                        KL1_sum, KL2_sum, JS_sum = KL1_sum + KL1, KL2_sum + KL2, JS_sum + JS
                        
                    self.train_hist['KL1_dist'].append(KL1_sum)
                    self.train_hist['KL2_dist'].append(KL2_sum)
                    self.train_hist['JS_dist'].append(JS_sum)
                                 
        return dLossErr, dLossReal, dLossFake, gLossErr, self.train_hist['KL1_dist'], self.train_hist['KL2_dist'], self.train_hist['JS_dist']

    def predict(self, ytest):
        noise = np.random.normal(0, 1, (ytest.shape[0], self.z_input_size))
        xpred = self.generator.predict([noise, ytest])
        return xpred

#     def sample(self, ytest, n_samples):
#         x_gan_samples = self.predict(ytest)
#         for i in range(n_samples - 1):
#             x_gan_pred = self.predict(ytest)
#             x_gan_samples = np.hstack([x_gan_samples, x_gan_pred])
#         return x_gan_samples
    def sample(self, ytest, n_samples):
        x_samples_gan = self.predict(ytest)
        for i in range(n_samples - 1):
            xpred_gan = self.predict(ytest)
            x_samples_gan = np.hstack([x_samples_gan, xpred_gan])
        return x_samples_gan


