from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_image(infilename):

    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32" )

    return data


def extract_patch(img):

    patches = []

    for i in range(0, rows):

        for j in range(0, columns):

            tmp = img[patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1)]

            patches.append(tmp.flatten())

    patches = np.array(patches)

    return patches.transpose()

# Part A

image = load_image('/q6/dataset/sad_days_gray.jpg')

img = np.mean(image, axis=2)

patch_size = 8

img_h = np.shape(img)[0]

img_w = np.shape(img)[1]

rows = np.shape(img)[0] // patch_size

columns = np.shape(img)[1] // patch_size

patchs = extract_patch(img)


# Part B

cov_matrix = np.cov(patchs)

eig_values, eig_vectors = np.linalg.eig(cov_matrix)

print(np.shape(eig_vectors))


def write_eigenvec_to_file(eig_values, file_path):

    f = open(file_path, "w+")

    for i in range(0, 20):
        f.write('eig val: ' + str(eig_values[i]) + '\neig vec:' + str(eig_vectors[i]))
        f.write('\n\n')

    f.close()

    return


# write_eigenvec_to_file(eig_values,"eigenval_eigenvec.txt" )
#
# mean_img = np.mean(patchs, axis=1)
#
# plt.imshow(np.reshape(mean_img, (8, 8)), cmap='gray')
#
# plt.title('mean_patch_fig')
#
# plt.show()

def plot_eigen_vectors(eig_vectors):

    print('came')
    for i in range(8):

        plt.imshow(np.reshape(eig_vectors[:, i], (8, 8)), cmap='gray')
        plt.title(str(i+1)+'-th eigen_vector')
        plt.show()
    print('done')
    return

# plot_eigen_vectors(eig_vectors)

# Part C

def compress(patchs, eigen_vectors,  mean_patch, size_):

    compressed_rep = []

    mean_ = np.reshape(mean_patch, (-1, 1))

    normalized_patches = np.subtract(patchs,  mean_)

    index = np.shape(eigen_vectors)[1]

    for i in range(0, index):

        vec = eigen_vectors[:,i]

        compressed_rep.append(np.matmul(np.reshape(vec, (1, size_)), normalized_patches))

    tmp = np.squeeze(compressed_rep)

    return tmp


# compressed_rep = compress(patchs, eig_vectors, mean_img, 64)
#
# eigen_2 = compressed_rep[:2, :]
#
# eigen_5 = compressed_rep[:5, :]
#
# eigen_10 = compressed_rep[:10, :]
#
# eigen_20 = compressed_rep[:20, :]


#  Part D

def reconstruct_image (compressed_rep, eigen_vectors, mean_img):

    tmp = np.reshape(mean_img, (-1, 1))

    image = np.matmul(eigen_vectors, compressed_rep) + tmp

    return image


# k2_recon = reconstruct_image(eigen_2, eig_vectors[:, :2] , mean_img)
#
# k5_recon = reconstruct_image(eigen_5, eig_vectors[:, :5] , mean_img)
#
# k10_recon = reconstruct_image(eigen_10, eig_vectors[:, :10] , mean_img)
#
# k20_recon = reconstruct_image(eigen_20, eig_vectors[:, :20] , mean_img)


# Part E

def unpatch(patches):

    new_image = np.zeros((img_h, img_w), dtype=np.float64)

    for i in range(0, rows):

        for j in range(0, columns):

            tmp = np.reshape(patches[:, i * columns + j], (patch_size, patch_size))

            new_image[patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1)] = tmp

    return new_image


def show_img(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.show()
    return


# recons_image_2 = unpatch(k2_recon)
#
# show_img(recons_image_2, 'Reconstructed by 2 eig_vec')
#
# recons_image_5 = unpatch(k5_recon)
#
# show_img(recons_image_5, 'Reconstructed by 5 eig_vec')
#
# recons_image_10 = unpatch(k10_recon)
#
# show_img(recons_image_10, 'Reconstructed by 10 eig_vec')
#
# recons_image_20 = unpatch(k20_recon)
#
# show_img(recons_image_20, 'Reconstructed by 20 eig_vec')