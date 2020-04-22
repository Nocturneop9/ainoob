from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False




pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
                                                     weights='imagenet')
pretrained_model.trainable = False

decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions




def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = image/255
  image = tf.image.resize(image, (224, 224))
  image = image[None, ...]
  return image

def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]




image_path = tf.keras.utils.get_file('D:\\bread.jpg', 'D:\\bread.jpg')
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)

image = preprocess(image)
image_probs = pretrained_model.predict(image)



plt.figure()
plt.imshow(image[0])
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()




loss_object = tf.keras.losses.BinaryCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  gradient = tape.gradient(loss, input_image)
  #signed_grad = tf.sign(gradient)
  return gradient



perturbations = create_adversarial_pattern(image, image_probs)
plt.imshow(perturbations[0])




def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0])
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()



epsilons = [0, 0.01, 0.1, 0.15]
descriptions = 'λ = 0.22, α = 1 ' 





# for i, eps in enumerate(epsilons):
#   adv_x = image + eps*perturbations
#   adv_x = tf.clip_by_value(adv_x, 0, 1)
#   display_images(adv_x, descriptions[i])

adv_x_pre = image
for j in range(0,50):
  perturbations = create_adversarial_pattern(adv_x_pre, image_probs)
  adv_x = adv_x_pre - (perturbations*255 +  0.22*(adv_x_pre - image))
  adv_x = tf.clip_by_value(adv_x, 0, 1)
  adv_x_pre = adv_x

display_images(adv_x, descriptions)

