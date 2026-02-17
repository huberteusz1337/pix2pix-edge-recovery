#Ten skrypt implementuje architekturę Pix2Pix GAN do treningu modelu 
#przekształcającego obrazy zakodowane z wysokim współczynnikiem kompresji (QP52)
#na obrazy z wykrytymi krawędziami (Canny edge detection).

#Główne cechy implementacji:
#    - Dynamiczny rozmiar wejścia generatora 
#    - Augmentacja danych (losowe przycinanie 128x128, odbicia lustrzane)
#    - Architektura U-Net z połączeniami skip dla generatora
#    - Dyskryminator PatchGAN


import tensorflow as tf
import os
import pathlib
import time
import datetime
import numpy as np
from matplotlib import pyplot as plt
from IPython import display

# --- Parametry eksperymentu ---
exp_steps=800000
save_weights_every_n_steps=10000 #dla generatora
save_ckpt_every_n_steps=200000 
show_train_progress_every_n_steps=2000

# --- Tworzenie katalogów na podstawie nazwy pliku ---
script_name = os.path.splitext(os.path.basename(__file__))[0]
exp_dir = f"exp_{script_name}"
test_dir = os.path.join(exp_dir, f"test_{script_name}")
train_dir = os.path.join(exp_dir, f"train_{script_name}")
ckpt_dir = os.path.join(train_dir, "ckpt")
weights_dir = os.path.join(train_dir, "weights")
progress_dir = os.path.join(exp_dir, f"training_progresss_{script_name}")
for d in [exp_dir, test_dir, train_dir, ckpt_dir, weights_dir, progress_dir]:
  os.makedirs(d, exist_ok=True)

# Parametry modelu i danych
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3

# Funkcja augmentacji 
def random_jitter(input_image, output_image):
  # Przykład: losowy flip
  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    output_image = tf.image.flip_left_right(output_image)
  return input_image, output_image

# Funkcja ładowania i normalizacji obrazów z PNG
def load_and_nor_image_for_training(input_path, output_path):
  input_image = tf.io.read_file(input_path)
  input_image = tf.io.decode_png(input_image, channels=3)
  output_image = tf.io.read_file(output_path)
  output_image = tf.io.decode_png(output_image, channels=3)
  input_image = tf.cast(input_image, tf.float32)
  output_image = tf.cast(output_image, tf.float32)
  input_image = (input_image / 127.5) - 1
  output_image = (output_image / 127.5) - 1
  return input_image, output_image

# Ścieżki do folderów treningowych
input_train_dir = pathlib.Path("/mnt/home/datasets/coco256/coco256_gt_in/train_30000_gt_in_qp52/")
output_train_dir = pathlib.Path("/mnt/home/datasets/coco256/coco256_gt_out/train_30000_gt_out_canny_less_edges_200_255/")
input_paths = sorted(input_train_dir.glob('*.png'))
output_paths = sorted(output_train_dir.glob('*.png'))
input_files = tf.data.Dataset.from_tensor_slices([str(p) for p in input_paths])
output_files = tf.data.Dataset.from_tensor_slices([str(p) for p in output_paths])
paired_files = tf.data.Dataset.zip((input_files, output_files))
train_dataset = paired_files.map(load_and_nor_image_for_training, num_parallel_calls=tf.data.AUTOTUNE)


# Przykład pobrania jednej pary do wizualizacji/generowania
example_input, example_target = next(iter(train_dataset.take(1)))



def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result



def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


def Generator():
  #inputs = tf.keras.layers.Input(shape=[256, 256, 3])
  inputs = tf.keras.layers.Input(shape=[None, None, 3])
  down_stack = [
    downsample(32, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(64, 4),  # (batch_size, 64, 64, 128)
    downsample(128, 4),  # (batch_size, 32, 32, 256)
    downsample(256, 4),  # (batch_size, 16, 16, 512)
    downsample(256, 4),  # (batch_size, 8, 8, 512)
    downsample(256, 4),  # (batch_size, 4, 4, 512)
    downsample(256, 4),  # (batch_size, 2, 2, 512)
  ]

  up_stack = [
    upsample(256, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(256, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(256, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(256, 4),  # (batch_size, 16, 16, 1024)
    upsample(128, 4),  # (batch_size, 32, 32, 512)
    upsample(64, 4),  # (batch_size, 64, 64, 256)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()

i_height = 128
i_width = 128

#make tensor (128,128,3)
example_input = tf.random.uniform(shape=(i_height, i_width, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)

#make tensor (1,128,128,3)
example_input = example_input[tf.newaxis, ...]


gen_output = generator(example_input, training=False)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(example_input[0] * 0.5 + 0.5)
plt.subplot(1,2,2)
plt.imshow(gen_output[0] * 0.5 + 0.5)
plt.savefig('obraz2')

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[128, 128, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[128, 128, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  

  down1 = downsample(32, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(64, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(128, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(256, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
#tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

disc_out = discriminator([example_input, gen_output], training=False)

#plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
#plt.colorbar()

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = ckpt_dir
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_images(model, test_input, tar, filename=None):
      
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))
  display_list = [test_input[0], prediction[0],tar[0]]
  title = ['Input Image', 'Predicted Image', 'Ground Truth Image']
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
 
  if filename is not None and progress_dir in filename:
    plt.savefig(filename)
  plt.close()
 


@tf.function
def train_step(input_image, target):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss



def get_augmented_pair(train_iterator):
    input, output = next(train_iterator)
    stacked = tf.stack([input, output], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, 128, 128, 3])
    input_cropped, output_cropped = cropped[0], cropped[1]
    # Losowy flip
    if tf.random.uniform(()) > 0.5:
        input_cropped = tf.image.flip_left_right(input_cropped)
        output_cropped = tf.image.flip_left_right(output_cropped)
    return input_cropped, output_cropped



def fit(train_ds, steps):
    start = time.time()
    train_iterator = iter(train_ds.repeat())
    step = 0

    loss_log_path = os.path.join(test_dir, "losses_log.csv")
    with open(loss_log_path, "w") as f:
      f.write("Step,gen_total_loss,gen_gan_loss,gen_l1_loss,disc_loss\n") 

    while step < exp_steps:
        input_batch, output_batch = [], []
        for _ in range(BATCH_SIZE):
            input_cropped, output_cropped = get_augmented_pair(train_iterator)
            input_batch.append(input_cropped)
            output_batch.append(output_cropped)
        
        input_batch = tf.stack(input_batch)
        output_batch = tf.stack(output_batch)
        gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_batch, output_batch)
        step += 1

        if (step + 1) % show_train_progress_every_n_steps == 0:
            filename = os.path.join(progress_dir, f'progress_step_{step+1}.png')
            generate_images(generator, input_batch, output_batch, filename=filename)
            print(f"\nSaved progress image at step {step+1}")

        if (step+1) % 100 == 0:
            print('.', end='', flush=True)

        if (step + 1) % save_ckpt_every_n_steps == 0:
            checkpoint.save(file_prefix=os.path.join(checkpoint_dir, f"ckpt_step_{step+1}"))

        if (step + 1) % save_weights_every_n_steps == 0:
            generator.save_weights(os.path.join(weights_dir, f'generator_weights_step_{step+1}.h5'))
            # Zapisz wartości funkcji strat do pliku tekstowego
            with open(loss_log_path, "a") as f:
                f.write(f"{step+1},{gen_total_loss.numpy()},{gen_gan_loss.numpy()},{gen_l1_loss.numpy()},{disc_loss.numpy()}\n")
        del gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss, input_batch, output_batch  # Zwolnij pamięć

    end = time.time()
    elapsed = end - start
    print(f"\nCzas treningu dla {exp_steps} kroków: {elapsed:.2f} sekund ({elapsed/60:.2f} minut)")
    
   
fit(train_dataset, steps=exp_steps)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
print("Trening zakonczony powodzeniem")










