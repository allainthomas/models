import datasets.fsns as fsns

DEFAULT_DATASET_DIR = '/content/models/research/attention_ocr/python/datasets/data/number_plates'

DEFAULT_CONFIG = {
    'name':
        'number_plates', # you can change the name if you want.
    'splits': {
        'train': {
            'size': 650, # change according to your own train-test split
            'pattern': 'train.tfrecord'
        },
        'test': {
            'size': 100, # change according to your own train-test split
            'pattern': 'test.tfrecord'
        }
    },
    'charset_filename':
        'charset-labels.txt',
    'image_shape': (200,100,3),#(max_width, max_height, 3),
    'num_of_views':
        1,
    'max_sequence_length':
        16, # TO BE CONFIGURED
    'null_code':
        133,
    'items_to_descriptions': {
        'image':
            'A 3 channel color image.',
        'label':
            'Characters codes.',
        'text':
            'A unicode string.',
        'length':
            'A length of the encoded text.',
        'num_of_views':
            'A number of different views stored within the image.'
    }
}


def get_split(split_name, dataset_dir=None, config=None):
  if not dataset_dir:
    dataset_dir = DEFAULT_DATASET_DIR
  if not config:
    config = DEFAULT_CONFIG

  return fsns.get_split(split_name, dataset_dir, config)