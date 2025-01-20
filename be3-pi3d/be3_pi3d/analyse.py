import cv2
import numpy as np
import matplotlib.pyplot as plt


def select_line(img, corners):
    """
    Retourne une matrice contenant 3 colonnes :
    - une pour les valeurs x de l'image,
    - une pour les valeurs y
    - une pour l'intensité de l'image en ce point
    """
    y_left = (corners[0, 1] + corners[3, 1]) // 2
    y_right = (corners[1, 1] + corners[2, 1]) // 2
    x_min = min(corners[0, 0], corners[1, 0])
    x_max = max(corners[0, 0], corners[1, 0])
    y = np.linspace(y_left, y_right, (x_max - x_min))
    line = np.zeros((x_max - x_min, 3))
    for i in range(line.shape[0]):
        x = x_min + i
        line[i, :] = [x, y[i], img[int(y[i]), x]]
    return line


def load_green_image(path_image, kernel_size: int = 50):
    """retourne le canal vert de l'image"""
    img = cv2.imread(path_image)

    if img is None:
        print("image not found : " + path_image)
        return None
    else:
        kernel2 = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
        # print("image loaded : " + path_image)
        img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)
        return img[:, :, 1]


def plot_all_images(imgs):
    # Afficher les images
    plt.figure()
    for n in range(imgs.shape[0]):
        plt.subplot(4, 4, n + 1)
        plt.imshow(imgs[n, :, :])
    plt.plot()


def plot_extracted_line(img, line):
    plt.figure()
    for p in line:
        print(p)
    plt.show()


# utile ????
def compute_cos_4_alphas(distances_axe_opt, focale):
    """Calcule alpha pour chaque distances à l'axe optique, à l'aide de p' = f'"""
    # CAHSOHTOA -> cos alpha = adj/hyp = f'/ Cp
    #           -> tan alpha = opp/adj = distances_axe_opt / f'
    # Or cos^2 = 1/(tan^2 +1)
    tan_alphas = distances_axe_opt / focale
    cos_4_alphas = 1 / (tan_alphas**2 + 1)
    return cos_4_alphas


def compute_nearly_E(cos_4_alphas, focale, lum):
    nearly_E = cos_4_alphas * lum / (focale**2)
    return nearly_E


def main() -> int:
    # Quelques variables
    corners = [
        [[1258, 1053], [1692, 1118], [3626, 2192], [1142, 2140]],
        [[1523, 1216], [3458, 1264], [3401, 2126], [1437, 2078]],
        [[1676, 1318], [3296, 1360], [3245, 2072], [1612, 2043]],
        [[1779, 1386], [3190, 1420], [3143, 2043], [1723, 2021]],
        [[1860, 1441], [3117, 1471], [3083, 2026], [1813, 2000]],
        [[1924, 1471], [3070, 1505], [3036, 2008], [1885, 1987]],
        [[1975, 1493], [3041, 1522], [3006, 1991], [1936, 1970]],
    ]
    corners = np.array(corners)

    path_images_folder = (
        "/home/flocondeglace/Documents/Ecole/PI3D/n7_BEPI3D/be3-pi3d/images/"
    )
    num_images = np.array([55] + list(np.arange(42, 55)))

    # Charger les images
    imgs = []
    for n in num_images:
        path_image = path_images_folder + "pi3d-" + str(n) + ".jpg"
        imgs.append(load_green_image(path_image))
    imgs = np.array(imgs)

    # Afficher des infos
    nb_images, height, width = imgs.shape
    print(f"There is {nb_images} images of size width={width}, height={height}")

    print(f"imgs shape : {imgs.shape}")

    # Trouver les lignes de l'image à analyser
    lines = []
    plt.figure()
    for i in range(corners.shape[0]):
        linei = select_line(imgs[i, :, :], corners[i, :, :])
        lines.append(linei)
        plt.plot(linei[:, 0], linei[:, 2])
    plt.ylabel("I_green")
    plt.xlabel("pixels")
    plt.show()

    plt.figure()
    for i in range(corners.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.plot(lines[i][:, 0], lines[i][:, 2])
        plt.ylabel("I")
        plt.xlabel("pixels")
    plt.show()

    return 0


if __name__ == "__main__":
    main()
