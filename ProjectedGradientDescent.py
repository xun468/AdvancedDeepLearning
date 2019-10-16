import tensorflow as tf

from utils import display_image


class ProjectedGradientDescent:
    def __init__(self, model, orig_image, target_label, exp=False):
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.iters = 60
        self.norm = 'l2'
        self.lr = 0.5
        self.eps = 30
        self.orig_image = orig_image
        self.model = model
        self.target_label = target_label

        self.exp = exp

        self.normalizer = 255   # TODO: Should we calculate the gradient in 0-1 or 0-255 image space...?
                                # Affects epsilon to a large degree
                                # It's gotta be in 0-1, because 30 eps in 0-256 wouldn't allow for any changes?
                                # But at the same time worse at actually changing a classification

    def step(self, image):
        with tf.GradientTape() as tape:
            tape.watch(image)
            pred = self.model(image)
            loss = self.loss(self.target_label, pred)

        grad = tape.gradient(loss, image)
        signed_grad = tf.sign(grad)

        image -= self.lr * signed_grad * self.normalizer

        diff = (image - self.orig_image) / self.normalizer  # Epsilon is measured in 256 image space

        if self.norm == 'l2':
            diff1 = tf.clip_by_norm(diff, self.eps)
            diff2 = clip_eta(diff, self.eps)

            # print(max(diff1 - diff2)) # Slight difference

            if not self.exp:
                diff = diff1
            else:
                diff = diff2

        elif self.norm == 'inf':
            diff = tf.clip_by_value(diff, -self.eps, self.eps)

        image = tf.clip_by_value(self.orig_image + diff * self.normalizer, 0, 255)

        return image

    def gen_adv_example(self, showImage=False):
        if showImage:
            display_image(self.orig_image, "Orig image")

        image = self.orig_image
        for i in range(self.iters):
            image = self.step(image)

        if showImage:
            display_image(image, "Orig image")

        return image


def clip_eta(eta, eps):
    # The 2-ball clipping from cleverhans, results in marginally different values than just tf.clip_by_norm()
    axis = list(range(1, len(eta.get_shape())))
    avoid_zero_div = 1e-12

    # avoid_zero_div must go inside sqrt to avoid a divide by zero in the gradient through this operation
    norm = tf.sqrt(tf.maximum(avoid_zero_div, tf.reduce_sum(tf.square(eta), axis, keepdims=True)))
    factor = tf.minimum(1., tf.math.divide(eps, norm))
    eta = eta * factor
    return eta
