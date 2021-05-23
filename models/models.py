from typing import Dict, List
from abc import ABC, abstractmethod
import time
import tensorflow as tf
from PIL import Image

class ModelStrategy(ABC):
    """
    The ModelStrategy interface declares operations common to all supported versions
    of some algorithm.

    The Model uses this interface to call the algorithm defined by Concrete ModelStrategies.

    """
    @abstractmethod
    def __init__(self, model_path: str):
        pass

    @abstractmethod
    def load_model(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, image_path: str):
        pass

    @abstractmethod
    def eval(self, model_path: str):
        pass

class Model():

    """
    The Model defines the interface of interest to clients.
    """

    def __init__(self, strategy: ModelStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> ModelStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: ModelStrategy) -> None:
        self._strategy = strategy

    def __call__(self, image):
        result = self._strategy.predict(image)
        print(result)
        return result

"""
Concrete ModelStrategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Model.

"""

class SH_Tranfer_Learning(ModelStrategy):

    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str) -> tf.keras.Model:

        print('Loading model...', end='')
        start_time = time.time()
        detect_fn = tf.saved_model.load(model_path)
        self.labels = read_label_map(model_path + "/label_map.pbtxt")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))

        return detect_fn

    def predict(self, image):
        input_tensor = tf.keras.preprocessing.image.img_to_array(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        prediction = self.model(input_tensor)
        # print(prediction)
        predicted_boxes = prediction['detection_boxes'].numpy().squeeze().tolist()
        # print(predicted_boxes)
        predicted_boxes = list(map(lambda x: x[:2][::-1] + x[2:][::-1],
                                   predicted_boxes))
        # print(predicted_boxes)
        predicted_labels = list(map(lambda x: self.labels[int(x)],list(prediction['detection_classes'].numpy().squeeze())))

        # print(predicted_labels)

        return {'boxes': predicted_boxes,
                'scores': prediction['detection_scores'].numpy().squeeze().tolist(),
                'labels': list(predicted_labels)}

    def eval(self, model_path: str):
        pass



class WD_Tranfer_Learning(ModelStrategy):

    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str) -> tf.keras.Model:

        print('Loading model...', end='')
        start_time = time.time()
        detect_fn = tf.saved_model.load(model_path)
        self.labels = read_label_map(model_path + "/label_map.pbtxt")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))

        return detect_fn

    def predict(self, image):
        input_tensor = tf.keras.preprocessing.image.img_to_array(image)
        input_tensor = input_tensor[tf.newaxis, ...]
        prediction = self.model(input_tensor)
        # print(prediction)
        predicted_boxes = prediction['detection_boxes'].numpy().squeeze().tolist()
        # print(predicted_boxes)
        predicted_boxes = list(map(lambda x: x[:2][::-1] + x[2:][::-1],
                                   predicted_boxes))
        # print(predicted_boxes)
        predicted_labels = list(map(lambda x: self.labels[int(x)],list(prediction['detection_classes'].numpy().squeeze())))

        # print(predicted_labels)

        return {'boxes': predicted_boxes,
                'scores': prediction['detection_scores'].numpy().squeeze().tolist(),
                'labels': list(predicted_labels)}

    def eval(self, model_path: str):
        pass

def read_label_map(label_map_path):
    item_id = None
    item_name = None
    items = {}
    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id:" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name:" in line:
                item_name = line.split(":", 1)[1].replace("\"", "").strip()
                
            if item_id is not None and item_name is not None:
                items[item_id] = item_name
                item_id = None
                item_name = None
    return items

if __name__ == '__main__':
    model = Model(WD_Tranfer_Learning('./wd_tranfer_learning'))
    image = Image.open("006.jpg")
    # print(image)
    model(image)

