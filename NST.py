import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# images
STYLE_IMAGE_PATH = "./Vassily_Kandinsky,_1913_-_Composition_7.jpg"
CONTENT_IMAGE_PATH = "./Oscar.jpg"

# constants
LR = 0.2
EPOCHS = 1000
IMAGE_SIZE = (396, 396)
CONTENT_LAYER = 9
CONTENT_WEIGHT = 0.33
STYLE_LAYERS = [1, 4, 7, 12]
STYLE_WEIGHTS = [0.125, 0.125, 0.5, 0.25]


def read_image(path):
    # read image and convert to tf tensor and resize
    max_dim = 512
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def deprocess(x):
    x *= 255
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def calc_content_loss(input_tensor):
    loss_value = loss(input_tensor, content_outputs)
    normalizer = 1.0
    return loss_value / (normalizer ** 2)


def loss(y, y_hat):
    return tf.reduce_mean(tf.math.squared_difference(y, y_hat))


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    normalizer = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(normalizer)


def calc_style_loss(target_outputs):
    style_loss = tf.add_n([tf.reduce_mean((style_output[k] - target_output[k])**2) * STYLE_WEIGHTS[k]
                           for k, v in enumerate(style_output)])
    return style_loss


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


if __name__ == "__main__":
    print(f"Runing TensorFlow version {tf.__version__}")

    # read in images
#    cv2.imshow("Style Image", cv2.imread(STYLE_IMAGE_PATH))
    style_image = read_image(STYLE_IMAGE_PATH)
    style_image = tf.image.resize(style_image, IMAGE_SIZE)

#    cv2.imshow("Content Image", cv2.imread(CONTENT_IMAGE_PATH))
    content_image = read_image(CONTENT_IMAGE_PATH)
    content_image = tf.image.resize(content_image, IMAGE_SIZE)

    # initialise image
    target = np.random.random((IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
#    cv2.imshow("Initialized Target", target)
    target = tf.image.convert_image_dtype(target, tf.float32)
    target = target[tf.newaxis, :]

    # THIS IS SO IMPORTANT !!
    target = tf.Variable(tf.cast(target, tf.float32))

    # load vgg16 and set layer
    vgg = VGG16(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                weights='imagenet', include_top=False)
    vgg.trainable = False
    print(vgg.summary())

    content_model = models.Model(
        inputs=vgg.inputs, outputs=vgg.layers[CONTENT_LAYER].output)
    print(f"Content layer {content_model.layers[CONTENT_LAYER].name}")

    names = [vgg.layers[l].name for l in STYLE_LAYERS]
    print(f"Style layers : {names}")
    style_model = models.Model(inputs=vgg.inputs,
                               outputs=[vgg.get_layer(name).output for name in names])

    # To save computational time, since this is static calculate only once
    content_outputs = content_model(content_image)
    style_outputs = style_model(style_image)
    style_output = [gram_matrix(output) for output in style_outputs]

    # define the optimizer
    opt = Adam(learning_rate=LR, beta_1=0.99, epsilon=1e-1)

    # Training loop
    losses = []
    grads = []
    print(f"Training on {EPOCHS} epochs")
    for i in range(1, EPOCHS):
        # get content loss
        with tf.GradientTape() as tape:
            target_output = style_model(target)
            target_output = [gram_matrix(output) for output in target_output]
            style_loss_value = calc_style_loss(target_output)

            target_output = content_model(target)
            content_loss_value = calc_content_loss(target_output)

            total_loss = CONTENT_WEIGHT * content_loss_value + \
                (1-CONTENT_WEIGHT) * style_loss_value

        losses.append(total_loss)

        # calculate gradient and update target
        grad = tape.gradient(total_loss, [target])
        grads.append(grad)
        opt.apply_gradients(zip(grad, [target]))
        target.assign(clip_0_1(target))

        # rule - reporting
        if i % 100 == 0:
            print(f"Iteration {i}:")
            print(f"\tContent Loss {np.round(content_loss_value.numpy(), 4)}")
            print(f"\tStyle Loss {np.round(style_loss_value.numpy(), 4)}")
            print(f"\tTotal loss {np.round(total_loss.numpy(), 4)}")

        if i % 1000 == 0:
            LR = LR / 10
            print(f"Learning Rate set to {LR}")
            opt = Adam(learning_rate=LR, beta_1=0.99, epsilon=1e-1)

    # show finalised image
    target = tf.convert_to_tensor(target).numpy()
    target = deprocess(target[0])
    print(f"target shape {target.shape}")
    cv2.imshow("Finalised Target", target)
    cv2.imwrite("./target.jpg", target)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.plot(range(len(losses)), losses, label="Losses from content")
    plt.title("Neural Style Transfer Loss")
    plt.ylim(0, np.max(losses)*1.1)
    plt.savefig("./NST.png")
    plt.show()

    print("** Done **")
