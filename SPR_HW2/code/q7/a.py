
from q4.functions import read_image
import os


def get_prior(path):

    directory = os.fsencode(path)
    tmp = 0
    length = 0

    for file in os.listdir(directory):

        filename = path+os.fsdecode(file)
        array = read_image(filename)
        tmp += sum(array)
        length += len(array)

    return 1-(tmp/length), (tmp/length)


prior_false, prior_true = get_prior('/q7/Dataset/Train/Masks/')

print(prior_false)
print(prior_true)


