from q4.functions import convert_image_to_array, get_center_of_images, classify

count = 2

directory = '/data/a/Train/'

train_images = convert_image_to_array(count, directory)

centers = get_center_of_images(train_images, count)

directory = '/data/a/Test/'
test_images = convert_image_to_array(count, directory)


classify(test_images, centers, count)