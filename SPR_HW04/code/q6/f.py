import matplotlib.pyplot as plt
import numpy as np

from q6.a_b_c_d_e import load_image, extract_patch, plot_eigen_vectors, compress, reconstruct_image, show_img


def unpatch_rgb(recon_patches):

    image = np.zeros((img_h, img_w, 3), dtype=np.float64)

    for i in range(rows):

        for j in range(columns):

            patch = np.reshape(recon_patches[:, i*columns+j], (patch_size, patch_size, 3))

            image[patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1)] = patch

    return image


image = load_image('/q6/dataset/sad_days_rgb.jpg')/255


patch_size = 8

img_h = np.shape(image)[0]

img_w = np.shape(image)[1]

rows = np.shape(image)[0] // patch_size

columns = np.shape(image)[1] // patch_size

patchs = extract_patch(image)

cov_matrix = np.cov(patchs)

eig_values, eig_vectors = np.linalg.eig(cov_matrix)


f = open("eigenval_eigenvec_f.txt","w+")

for i in range(0, 20):

    f.write('eig val: '+str(eig_values[i])+'\neig vec:'+ str(eig_vectors[i]))
    f.write('\n\n')

f.close()

def normalize_array(image):

    imin = np.min(image)
    imax = np.max(image)
    out_img = (image-imin) / (imax-imin)

    return out_img

def plot_eigen_vectors_rgb(vec):

    for i in range(8):
        new_shape = np.reshape(vec[:, i], (8, 8, 3))
        plt.imshow(normalize_array(new_shape))
        plt.title(str(i+1)+'-th eigen_vector')
        plt.show()

    return


mean_img = np.mean(patchs, axis=1)

new_shape = np.reshape(patchs, (8, 8, 3))

plt.imshow(normalize_array(new_shape))

plt.title('mean_patch_fig')

plt.show()

plot_eigen_vectors_rgb(eig_vectors)

compressed_rep = compress(patchs, eig_vectors, mean_img, 64*3)

eigen_2 = compressed_rep[:2, :]

eigen_5 = compressed_rep[:5, :]

eigen_10 = compressed_rep[:10, :]

eigen_20 = compressed_rep[:20, :]

k2_recon = reconstruct_image(eigen_2, eig_vectors[:, :2] , mean_img)

k5_recon = reconstruct_image(eigen_5, eig_vectors[:, :5] , mean_img)

k10_recon = reconstruct_image(eigen_10, eig_vectors[:, :10] , mean_img)

k20_recon = reconstruct_image(eigen_20, eig_vectors[:, :20] , mean_img)

recons_image_2 = unpatch_rgb(k2_recon)

show_img(recons_image_2, 'Reconstructed by 2 eig_vec')

recons_image_5 = unpatch_rgb(k5_recon)

show_img(recons_image_5, 'Reconstructed by 5 eig_vec')

recons_image_10 = unpatch_rgb(k10_recon)

show_img(recons_image_10, 'Reconstructed by 10 eig_vec')

recons_image_20 = unpatch_rgb(k20_recon)

show_img(recons_image_20, 'Reconstructed by 20 eig_vec')