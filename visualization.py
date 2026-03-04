import matplotlib.pyplot as plt

def vec2img(row) : 
    """
    Transform a vector of data to a image of shape (32,32,3) 
    """
    red = row[0:1024].reshape(32, 32)
    green = row[1024:2048].reshape(32, 32)
    blue = row[2048:3072].reshape(32, 32)

    img = np.dstack((red,green,blue))

    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min)

    return img

def plot_img(img,label) :
    """
    Plot a image 
    """
    plt.figure()
    plt.imshow(img)
    plt.title(f"Label {label}")
    plt.show()

