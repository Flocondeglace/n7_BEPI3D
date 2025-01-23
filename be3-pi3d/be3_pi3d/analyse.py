import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import rawpy
import exifread


def func(x, a, b, c):
    """
    Fonction d'interpolation pour lisser les courbes.
    """
    return a * x**2 + b * x + c


def select_line(img, corners):
    """
    Cette fonction retourne une matrice contenant 4 colonnes :
    - une pour les valeurs x de l'image,
    - une pour les valeurs y
    - une pour l'intensité de l'image en ce point
    - une pour approx

    Arguments:
        img (np.ndarray): l'image dont on veut extraire les informations
        corners (list): liste des coordonnées des QRCode

    Return:
        line (np.ndarray): 
        popt (array): tableau de valeurs pour ajuster la fonction func à des données
    """
    y_left = (corners[0, 1] + corners[3, 1]) // 2
    y_right = (corners[1, 1] + corners[2, 1]) // 2
    x_min = min(corners[0, 0], corners[1, 0])
    x_max = max(corners[0, 0], corners[1, 0])
    y = np.linspace(y_left, y_right, (x_max - x_min))
    line = np.zeros((x_max - x_min, 4))
    for i in range(line.shape[0]):
        x = x_min + i
        line[i, :] = [x, y[i], img[int(y[i]), x], 0]
    popt, pcov = scipy.optimize.curve_fit(func, line[:, 0], line[:, 2]) # Utiliser les moindres carrés non linéaires pour ajuster une fonction, f, à des données
    line[:, 3] = func(line[:, 0], *popt)
    return line, popt


def load_image(path_image, kernel_size: int = 20, plot=False):
    """
    Cette fonction retourne le canal vert de l'image lissée.
    
    Arguments:
        path_image (str): chemin de l'image .raw à charger
        kernel_size (int): taille du noyau pour le lissage de l'image (enlever les granules sur le fond vert)
        plot (bool): Afficher l'image .raw

    Return:
        img (np.ndarray): le canal vert de l'image .raw lissée

    """
    raw = rawpy.imread(path_image)

    if raw is None:
        print("image not found : " + path_image)
        return None
    else:
        # Pour récupérer la focale
        # file = open(path_image, "rb")
        # tags = exifread.process_file(file)
        # print("focale : ", tags["EXIF FocalLength"])
        # print([str(k) for k in tags.keys()])

        raw_data = raw.raw_image_visible
        pattern = raw.raw_pattern
        # print("pattern : ", pattern)
        mask = pattern % 2 == 1
        green_mask = np.tile(mask, (raw_data.shape[0] // 2, raw_data.shape[1] // 2))
        img = np.where(green_mask, raw_data, 0)
        # Interpoler le vert des autres pixels
        filter_green_kernel = np.array(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32
        )
        filter_green_kernel /= filter_green_kernel.sum()
        # Calculer la moyenne locale des pixels verts pour remplacer les pixels vide
        mean_values = cv2.filter2D(
            img.astype(np.float32), ddepth=-1, kernel=filter_green_kernel
        )
        zero_positions = img == 0
        img[zero_positions] = mean_values[zero_positions]

        # Afficher les raw
        if plot:
            plt.figure()
            plt.imshow(img, cmap="gray")
            plt.show()

        # Lisser l'image
        kernel_lissage = np.ones((kernel_size, kernel_size), np.float32) / (
            kernel_size**2
        )
        # print("image loaded : " + path_image)
        img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel_lissage)
        return img


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


def get_corners(imgs, filename):
    corners = np.zeros((imgs.shape[0], 4, 2), dtype=np.int64)
    print(corners.shape)
    for i in range(imgs.shape[0]):
        img = imgs[i, :, :]
        plt.imshow(img)
        plt.title("Cliquez sur les 4 coins intérieurs")
        plt.axis("on")
        print("Veuillez cliquer sur 4 points sur l'image.")
        points = plt.ginput(4)
        plt.close()
        corners[i, :, :] = np.array(points, dtype=np.int64)
    print(corners)
    # np.savetxt(filename, corners, delimiter=",")
    np.save(filename, corners)
    print(f"Corners saved in {filename}")
    return corners


def main() -> int:
    distance_2_arukos_width = 40.2  # cm
    distance_first_cam = 60.0  # cm

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
    corners = [
        [[1292.0, 1052.0], [3688.0, 1119.0], [3582.0, 2192.0], [1139.0, 2144.0]],
        [[1503.0, 1215.0], [3438.0, 1292.0], [3362.0, 2106.0], [1427.0, 2087.0]],
        [[1676.0, 1349.0], [3266.0, 1368.0], [3209.0, 2068.0], [1609.0, 2058.0]],
        [[1771.0, 1397.0], [3151.0, 1445.0], [3113.0, 2049.0], [1743.0, 2001.0]],
        [[1886.0, 1435.0], [3094.0, 1445.0], [3113.0, 1991.0], [1810.0, 1991.0]],
        [[1915.0, 1464.0], [3046.0, 1531.0], [3007.0, 2020.0], [1886.0, 1991.0]],
        [[1944.0, 1493.0], [3036.0, 1541.0], [3017.0, 1982.0], [1925.0, 1962.0]],
        [[2021.0, 1512.0], [3007.0, 1531.0], [2969.0, 1972.0], [2001.0, 1943.0]],
        [[2059.0, 1522.0], [2959.0, 1570.0], [2969.0, 1953.0], [2049.0, 1962.0]],
        [[2107.0, 1541.0], [2969.0, 1570.0], [2921.0, 1953.0], [2078.0, 1905.0]],
        [[2135.0, 1550.0], [2931.0, 1598.0], [2912.0, 1924.0], [2088.0, 1905.0]],
        [[2212.0, 1560.0], [2902.0, 1598.0], [2892.0, 1905.0], [2116.0, 1895.0]],
        [[2174.0, 1589.0], [2931.0, 1598.0], [2854.0, 1905.0], [2174.0, 1905.0]],
        [[2212.0, 1589.0], [2883.0, 1627.0], [2864.0, 1905.0], [2222.0, 1867.0]],
    ]

    corners = np.array(corners, dtype=np.int64)

    # path_images_folder = (
    #     "/home/flocondeglace/Documents/Ecole/PI3D/n7_BEPI3D/be3-pi3d/images/"
    # )
    path_images_folder = (
        "/home/jureme/PI3D/n7_BEPI3D/be3-pi3d/images/"
    )
    

    num_images = np.array([55] + list(np.arange(42, 55)))

    # Charger les images
    # imgs = []
    # for n in num_images:
    #     path_image = path_images_folder + "pi3d-" + str(n) + ".cr2"
    #     imgs.append(load_image(path_image))
    # imgs = np.array(imgs)
    imgs = []
    for path_image in os.listdir(path_images_folder):      
        imgs.append(load_image( path_images_folder + path_image))
    imgs = np.array(imgs)

    # path_corners = (
    #     "/home/flocondeglace/Documents/Ecole/PI3D/n7_BEPI3D/be3-pi3d/corners.npy"
    # )
    # path_corners = (
    #     "/home/jureme/PI3D/n7_BEPI3D/be3-pi3d/corners.npy"
    # )
    # # Corners from file
    # if os.path.isfile(path_corners):
    #     # corners = np.loadtxt(path_corners, delimiter=",")
    #     corners = np.load(path_corners)
    #     corners = np.array(corners, dtype=np.int64)
    # else:
    #     corners = get_corners(imgs, path_corners)

    # Afficher des infos
    nb_images, height, width = imgs.shape
    print(f"There is {nb_images} images of size width={width}, height={height}")
    # print(f"Corners : {corners}")

    print(f"imgs shape : {imgs.shape}")

    # Trouver les lignes de l'image à analyser
    lines = []
    taille_pixels = []
    alphas = []
    plt.figure()
    step = 2
    for i in range(0, corners.shape[0], step):
        linei, popti = select_line(imgs[i, :, :], corners[i, :, :])
        lines.append(linei)
        x_axis = (linei[:, 0] - linei[0, 0]) / (linei[-1, 0] - linei[0, 0])
        nb_pixels_ligne = len(linei[:, 3])
        current_taille_pixels = distance_2_arukos_width / nb_pixels_ligne
        taille_pixels.append(current_taille_pixels)
        distance_cami = current_taille_pixels * distance_first_cam / taille_pixels[0]
        alphas.append(
            (2 / np.sqrt((x_axis * current_taille_pixels) ** 2 + 2**2)) ** 4
            # compute_cos_4_alphas(
            #    np.sqrt((x_axis * current_taille_pixels) ** 2 + distance_cami**2), 2
            # )
        )

        print(f"distance : {distance_cami} cm")
        plt.plot(x_axis, linei[:, 3])
        plt.text(
            x_axis[-1], linei[-1, 3], str(num_images[i]), fontsize=10, color="black"
        )
        # print(f"paramètre de la droite de l'image {num_images[i]} : {popti}")
    plt.legend([str(num_images[i]) for i in range(0, corners.shape[0], step)])
    plt.ylabel("I")
    plt.xlabel("pixels")
    plt.show()

    # experience alpha
    # plt.figure()
    # for i in range(3):  # len(alphas)):
    #     plt.plot(alphas[i] / max(alphas[i]))
    # plt.legend([str(i) for i in range(0, 3)])
    # plt.show()

    # Plot interpolation
    plt.figure()
    for i in range(len(lines)):
        plt.subplot(4, 4, i + 1)
        plt.plot(lines[i][:, 0], lines[i][:, 2])
        plt.plot(lines[i][:, 0], lines[i][:, 3])
        plt.ylabel("I")
        plt.xlabel("distance (cm)")
        plt.title(str(num_images[i * step]))
    plt.tight_layout()
    plt.show()
    plt.figure()
    plt.plot(lines[0][:, 3])
    plt.plot(
        lines[-1][:, 3] * (np.cos(lines[-1][:, 0] * taille_pixels[-1])) ** 4 + 2000
    )
    plt.legend(["normal", "inventer"])
    plt.show()

    return 0


if __name__ == "__main__":
    main()
