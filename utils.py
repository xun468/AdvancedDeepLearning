import matplotlib.pyplot as plt


def display_image(image, description=''):
    if len(image.shape) == 4:
        plt.imshow(image[0])
    else:
        plt.imshow(image)
    plt.title(description)
    plt.axis('off')
    plt.show()
