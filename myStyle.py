import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

import matplotlib.pyplot as plt
import numpy as np
import time


# images
CONTENT_IMAGE = "./Oscar.jpg"
STYLE_IMAGE = "./Vassily_Kandinsky,_1913_-_Composition_7.jpg"

# constants
EPOCHS = 500
LR = 0.02

CONTENT_LAYERS = ["block5_conv2"]
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']

IMAGE_SIZE = (224, 224)
STYLE_WEIGHT = 1e-2
CONTENT_WEIGHT = 1e4

num_content_layers = len(CONTENT_IMAGE)
num_style_layers = len(STYLE_IMAGE)


def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img, IMAGE_SIZE)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None, save=False):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

    if save:
        plt.savefig(f"./NST_{title}.png")

    plt.show()


def calc_content_loss(content_outputs):
    return tf.reduce_mean(tf.math.squared_difference(content_targets, content_outputs)) / num_content_layers


def calc_style_outputs(inputs):
    preprocessed_input = preprocess_input(inputs)
    outputs = style_model(preprocessed_input)
    style_outputs = [gram_matrix(style_output) for style_output in outputs]
    return style_outputs


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / (num_locations)


def calc_style_loss(outputs):
    style_loss = tf.add_n([tf.reduce_mean((outputs[k] - style_targets[k])**2)
                           for k, v in enumerate(outputs)])
    return style_loss / num_style_layers


def calc_total_loss(style_loss, content_loss):
    return STYLE_WEIGHT * style_loss + CONTENT_WEIGHT * content_loss


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def train_step():
    print(f"Training on {EPOCHS} epochs")
    content_losses, style_losses, total_losses = [], [], []
    grads = []

    # set starting image - use content image to save time
    image = tf.Variable(content_image)

    for n in range(EPOCHS):
        with tf.GradientTape() as tape:
            # get content loss
            outputs = content_model(image * 255)
            content_loss = calc_content_loss(outputs)
            # get style loss
            outputs = calc_style_outputs(image * 255)
            style_loss = calc_style_loss(outputs)
            total_loss = calc_total_loss(style_loss, content_loss)

        content_losses.append(content_loss)
        style_losses.append(style_loss)
        total_losses.append(total_loss)

        grad = tape.gradient(total_loss, image)
        grads.append(grad)

        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

        # rule - reporting
        if n % 100 == 0:
            print(f"Iteration {n}:")
            print(f"\tContent Loss {content_loss:.4f}")
            print(f"\tStyle Loss {style_loss:.4f}")
            print(f"\tTotal loss {total_loss:.4f}")

    losses = [content_losses, style_losses, total_losses]
    return image, losses, grads


def print_losses(losses, title=None):
    titles = ["Content Losses", "Style Losses", "Total Losses"]
    for i in range(3):
        x = losses[i]

        plt.plot(range(len(x)), x, label=titles[i])
        plt.title(f"Neural Style Transfer:  {titles[i]}")
        plt.ylim(0, np.max(x) * 1.1)
        plt.savefig(f"./NST_{titles[i]}.png")
        plt.show()


if __name__ == "__main__":
    content_image = load_img(CONTENT_IMAGE)
    style_image = load_img(STYLE_IMAGE)

    imshow(content_image, 'Content Image')
    imshow(style_image, 'Style Image')

    vgg = VGG16(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                include_top=False, weights='imagenet')
    vgg.trainable = False
    print(vgg.summary())

    # define content target
    content_model = models.Model(inputs=vgg.inputs,
                                 outputs=[vgg.get_layer(name).output for name in CONTENT_LAYERS])
    print(f"Content layer : {CONTENT_LAYERS}")

    # define style target
    style_model = models.Model(inputs=vgg.inputs,
                               outputs=[vgg.get_layer(name).output for name in STYLE_LAYERS])
    print(f"Style layers : {STYLE_LAYERS}")

    # set targets
    content_targets = content_model(content_image * 255)
    style_targets = calc_style_outputs(style_image * 255)

    # set optimizer
    opt = tf.optimizers.Adam(learning_rate=LR)

    # training
    start = time.time()
    image, losses, grads = train_step()
    end = time.time()
    print(f"Total time: {end-start:.1f}")
    imshow(image, "Final image", True)
    print_losses(losses, "Losses")

    print("** Done **")
