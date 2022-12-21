import matplotlib.pyplot as plt

main_path = './dataset/'
# classes = [  'daisy', 'dandelion','rose','sunflower','tulip']

img_size=[64, 64]
batch_size = 32

from keras.utils import image_dataset_from_directory

ulaz_trening = image_dataset_from_directory(main_path,
                                            subset='training',
                                            validation_split=0.2,
                                            seed=52,
                                            batch_size= batch_size,
                                            image_size=img_size)

ulaz_test=image_dataset_from_directory(main_path,
                                       subset='validation',
                                       validation_split=0.2,
                                       seed=52,
                                       batch_size= batch_size)


classes = ulaz_trening.class_names


import matplotlib.pyplot as plt
for img, lab in ulaz_trening.take(1):
    plt.figure()
    for k in range(10):
        plt.subplot(2, 5, k+1)
        plt.imshow(img[k].numpy().astype('uint8'))
        plt.title(classes[lab[k]])
    plt.show()


from keras import Sequential
from keras import layers

model = Sequential([
    layers.Conv2D(16, 3, activation='relu', input_shape=(64, 64, 3), padding='same'),
    layers.MaxPooling2D(2, strides=2),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2, strides=2),

    layers.Conv2D(64, 3, activation='relu', padding='same'),
    layers.MaxPooling2D(2, strides=2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dense(len(classes), 'softmax')
])

model.summary()

