#abstract class
from abc import ABC, abstractmethod

#load_tfod_api
import tensorflow as tf
from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np

class Model(ABC):

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def __call__(self):
        pass


class Context():

    def __init__(self, strategy: Model) -> None:

        self._strategy = strategy

    @property
    def strategy(self) -> Model:

        self._strategy

    @strategy.setter
    def strategy(self, strategy: Model) -> None:
        
        self._strategy = strategy

    def __call__(self, image_path) -> None:
        return self._strategy(image_path)



class DetectWord(Model):
    def __init__(self) -> None:
        self._load_model()

    def _load_model(self):
        configs = config_util.get_configs_from_pipeline_file("detect_word/pipeline.config")
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore("detect_word/checkpoint/ckpt-50").expect_partial()
        self.model = detection_model

    def _detect_fn(self,image):
        image, shapes = self.model.preprocess(image)
        prediction_dict = self.model.predict(image, shapes)      
        detections = self.model.postprocess(prediction_dict, shapes)
        return detections

    def __call__(self, image):

        category_index = label_map_util.create_category_index_from_labelmap("detect_word/label_map.pbtxt")
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        detections = self._detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.1,
                    agnostic_mode=False)

        return image_np_with_detections

class DetectSheet(Model):
    def __init__(self) -> None:
        self._load_model()

    def _load_model(self):
        configs = config_util.get_configs_from_pipeline_file("detect_sheet/pipeline.config")
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)

        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore("detect_sheet/checkpoint/ckpt-50").expect_partial()
        self.model = detection_model

    def _detect_fn(self,image):
        image, shapes = self.model.preprocess(image)
        prediction_dict = self.model.predict(image, shapes)      
        detections = self.model.postprocess(prediction_dict, shapes)
        return detections

    def __call__(self, image):

        category_index = label_map_util.create_category_index_from_labelmap("detect_sheet/labelmap.pbtxt")
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
        detections = self._detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.1,
                    agnostic_mode=False)

        return image_np_with_detections
