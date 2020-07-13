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
LR = 0.01
HIDDEN_LAYER = 9
THRESHOLD = 50
EPOCHS = int(50000 * (HIDDEN_LAYER/3))

STYLE_IMAGE_PATH = "./style.jpg"
CONTENT_IMAGE_PATH = "./skyline2.jpg"


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


def loss(y, y_hat):
    return tf.reduce_mean(tf.math.squared_difference(y, y_hat))


if __name__ == "__main__":
    print(f"Runing TensorFlow version {tf.__version__}")

    #read in images
    y = read_image(CONTENT_IMAGE_PATH)
    cv2.imshow("Content Image", cv2.imread(CONTENT_IMAGE_PATH))

    # initialise image
    target = np.random.random((y.shape[1], y.shape[2], 3))
    cv2.imshow("Initialized Target", target)
    target = np.expand_dims(target, axis=0)
    target = preprocess_input(target)
    print(f"Target image shape {target.shape}")

    # THIS IS SO IMPORTANT !!
    target = tf.Variable(tf.cast(target, tf.float32))

    # load vgg16 and set layer
    vgg = VGG16(input_shape=(y.shape[1], y.shape[2], 3),
                weights='imagenet', include_top=False)
    model = models.Model(inputs=vgg.inputs,
                         outputs=vgg.layers[HIDDEN_LAYER].output)
    print(model.summary())

    # define the optimizer
    opt = Adam(lr=LR, decay=LR / EPOCHS)

    # Training loop
    losses = []
    print(f"Training on {EPOCHS} epochs")
    for i in range(EPOCHS):
        # get content loss
        with tf.GradientTape() as tape:
            loss_value = loss(model(y), model(target))

        losses.append(loss_value)
        if loss_value < THRESHOLD:
            break

        # calculate gradient and update target
        grad = tape.gradient(loss_value, [target])
        opt.apply_gradients(zip(grad, [target]))

        if i % 500 == 0:
            print(f"Iteration {i}\tLoss {loss_value.numpy()}")

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
