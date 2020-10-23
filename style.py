import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

# constants
IMAGE_SIZE = (224, 224)
STYLE_LAYERS = [1, 4, 7, 12, 17]
STYLE_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
LR = 100
EPOCHS = 100
STYLE_IMAGE_PATH = "./Vassily_Kandinsky,_1913_-_Composition_7.jpg"


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
    print(f"\n\nRunning TensorFlow version {tf.__version__}")

    # read in images
#    cv2.imshow("Style Image", cv2.imread(STYLE_IMAGE_PATH))
    style_image = read_image(STYLE_IMAGE_PATH)
#    style_image = preprocess_input(style_image*255.)
    style_image = tf.image.resize(style_image, IMAGE_SIZE)

    # initialise image
    target = np.random.random((style_image.shape[1], style_image.shape[2], 3))
#    cv2.imshow("Initialized Target", target)
    target = tf.image.convert_image_dtype(target, tf.float32)
    target = target[tf.newaxis, :]
 #   target = preprocess_input(target*255.)

    # THIS IS SO IMPORTANT !!
    target = tf.Variable(tf.cast(target, tf.float32))

    # load vgg16 and set layer
    vgg = VGG19(weights='imagenet', include_top=False)
    vgg.trainable = False
    print(vgg.summary())

    names = [vgg.layers[l].name for l in STYLE_LAYERS]
    print(f"Style layers : {names}")
    style_model = models.Model(inputs=vgg.input,
                               outputs=[vgg.get_layer(name).output for name in names])

    # To save computational time, since this is static calculate only once
    style_outputs = style_model(style_image)
    style_output = [gram_matrix(output) for output in style_outputs]

    # Define optimizer
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
            loss_value = calc_style_loss(target_output)

        losses.append(loss_value.numpy())

        # calculate gradient and update target
        grad = tape.gradient(loss_value, [target])
        grads.append(grad)
        opt.apply_gradients(zip(grad, [target]))
        target.assign(clip_0_1(target))

        # rule - reporting
        if i % 100 == 0:
            print(f"Iteration {i}\tLoss {loss_value.numpy()}")

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
    plt.savefig("./NST_style_loss.png")
    plt.show()

    print("** Done **")
