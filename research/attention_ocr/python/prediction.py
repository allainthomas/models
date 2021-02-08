"""
Usage:
python prediction.py --checkpoint=/content/ocr_model/ocr_model/model.ckpt-6000\
  --image_path_pattern=/content/models/custom_generator/data/crops/ --dataset_name=number_plates
  """
import numpy as np
import PIL.Image
import os

import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.training import monitored_session

import common_flags
import datasets
import data_provider

FLAGS = flags.FLAGS
common_flags.define()

# e.g. ./datasets/data/fsns/temp/fsns_train_%02d.png
flags.DEFINE_string('image_path_pattern', '',
                    'A file pattern with a placeholder for the image index.')


def get_dataset_image_size(dataset_name):
  # Ideally this info should be exposed through the dataset interface itself.
  # But currently it is not available by other means.
  ds_module = getattr(datasets, dataset_name)
  height, width, _ = ds_module.DEFAULT_CONFIG['image_shape']
  return width, height


def load_images(img_paths, batch_size, dataset_name):
  width, height = get_dataset_image_size(dataset_name)
  images_actual_data = np.ndarray(shape=(batch_size, height, width, 3),
                                  dtype='uint8')

  for i in range(batch_size):
    path = img_paths[i]
    pil_image = PIL.Image.open(tf.io.gfile.GFile(path, 'rb')).resize((100,200))
    #pil_image = PIL.Image.open(tf.io.gfile.GFile(path, 'rb'))
    images_actual_data[i, ...] = np.asarray(pil_image)
  return images_actual_data


def create_model(batch_size, dataset_name):
  width, height = get_dataset_image_size(dataset_name)
  dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
  model = common_flags.create_model(
      num_char_classes=dataset.num_char_classes,
      seq_length=dataset.max_sequence_length,
      num_views=dataset.num_of_views,
      null_code=dataset.null_code,
      charset=dataset.charset)
  raw_images = tf.compat.v1.placeholder(
      tf.uint8, shape=[batch_size, height, width, 3])
  images = tf.map_fn(data_provider.preprocess_image, raw_images,
                     dtype=tf.float32)
  endpoints = model.create_base(images, labels_one_hot=None)
  return raw_images, endpoints


def run(checkpoint, batch_size, dataset_name, images_path):
  # Create model
  images_placeholder, endpoints = create_model(batch_size,
                                               dataset_name)

  # Load pre-trained model
  session_creator = monitored_session.ChiefSessionCreator(
       checkpoint_filename_with_path=checkpoint)

  # Find images
  img_names = os.listdir(images_path)
  img_names.sort()
  print("\nNumber of images to process : ", len(img_names))
  img_paths = [images_path + img_name for img_name in img_names]
  print("Number of images paths : ", len(img_paths))
  global_results = []
  with monitored_session.MonitoredSession(session_creator=session_creator) as sess:
    # Loop per batch of size 1
    for i, img_path in enumerate(img_paths):
        print("\nNew Image :", img_path)

        images_data = load_images([img_path], batch_size,
                            dataset_name)

        predictions = sess.run(endpoints.predicted_text,
                           feed_dict={images_placeholder: images_data})
        result  = [pr_bytes.decode('utf-8') for pr_bytes in predictions.tolist()]
        for line in result :
            print(result)
            global_results.append(line)
        print("Image :", i)
  return global_results


def main(_):
  print("Predicted strings:")
  predictions = run(FLAGS.checkpoint, FLAGS.batch_size, FLAGS.dataset_name,
                    FLAGS.image_path_pattern)
  print("\n End of processing")
  print("\n Got ", len(predictions), " predictions ")
  for line in predictions:
    print(line)


if __name__ == '__main__':
  tf.compat.v1.app.run()
