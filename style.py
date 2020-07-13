import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# constants
LR = 10.0
STYLE_LAYERS = [1, 4, 7, 11]
STYLE_WEIGHTS = [0.25, 0.125, 0.125, 0.5]
THRESHOLD = 0.0050
EPOCHS = 5000
STYLE_IMAGE_PATH = "./style.jpg"


def read_image(path):
    img = image.load_img(path, target_size=None)
    # convert style image to array and preprocess for vgg
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(f"Shape of {path} image is {x.shape}")
    return x


def deprocess(x):
    x -= x.mean()
    x /= x.std() + 1e-05
    x *= 0.1 * x

    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype("uint8")
    return x


def calc_style_loss(x1, x2):
    losses = []
    for k, _ in enumerate(x1):
        x = x1[k]
        x = tf.squeeze(x)
        m1 = tf.reshape(x,
                        shape=(tf.shape(x)[2], tf.shape(x)[0] * tf.shape(x)[1]))

        y = x2[k]
        y = tf.squeeze(y)
        m2 = tf.reshape(y,
                        shape=(tf.shape(y)[2], tf.shape(y)[0] * tf.shape(y)[1]))

        gram1 = tf.matmul(m1, tf.transpose(m1))
        gram2 = tf.matmul(m2, tf.transpose(m2))

        normalizer = 2 * tf.shape(y)[0] * tf.shape(y)[1] * tf.shape(y)[2]
        normalizer = tf.cast(normalizer, tf.float32)

        layer_loss = loss(gram1, gram2) / (normalizer ** 2)

        layer_loss = STYLE_WEIGHTS[k] * layer_loss

        losses.append(layer_loss)

    total_loss = tf.reduce_sum(losses)

    return total_loss


def loss(y, y_hat):
    return tf.reduce_mean(tf.math.squared_difference(y, y_hat))


if __name__ == "__main__":
    print(f"\n\nRunning TensorFlow version {tf.__version__}")

    # read in images
    x = read_image(STYLE_IMAGE_PATH)
    cv2.imshow("Style Image", cv2.imread(STYLE_IMAGE_PATH))

    # initialise image
    target = np.random.random((x.shape[1], x.shape[2], 3))
    cv2.imshow("Initialized Target", target)
    target = np.expand_dims(target, axis=0)
    target = preprocess_input(target)
    print(f"Target image shape {target.shape}")

    # THIS IS SO IMPORTANT !!
    target = tf.Variable(tf.cast(target, tf.float32))

    # load vgg16 and set layer
    vgg = VGG16(input_shape=(x.shape[1], x.shape[2], 3),
                weights='imagenet', include_top=False)
    vgg.trainable = False
    print(vgg.summary())

    names = [vgg.layers[l].name for l in STYLE_LAYERS]
    print(f"Style layers : {names}")
    style_model = models.Model(inputs=vgg.inputs,
                               outputs=[vgg.get_layer(name).output for name in names])

    # Define optimizer
    opt = Adam(learning_rate=LR, decay=LR / EPOCHS)

    # Training loop
    losses = []
    grads = []
    print(f"Training on {EPOCHS} epochs")
    for i in range(EPOCHS):
        # get content loss
        with tf.GradientTape() as tape:
            loss_value = calc_style_loss(style_model(x), style_model(target))

        losses.append(loss_value)

        # calculate gradient and update target
        grad = tape.gradient(loss_value, [target])
        grads.append(grad)
        opt.apply_gradients(zip(grad, [target]))

        # rule - stopping
        if i > 1 and loss_value < THRESHOLD:
            break

        # rule - reporting
        if i % 100 == 0:
            print(f"Iteration {i}\tLoss {loss_value.numpy()}")

        # rule - update learning rate
        if i > 0 and i % 1000 == 0:
            LR /= 10
            print(f"Learning Rate reduced to {LR}")

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
    plt.show()
    plt.savefig("./NST_content_loss_{HIDDEN_LAYER}.png")

    print("** Done **")
