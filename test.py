import sys
from models.ModelManager import ModelManager

if __name__ == '__main__':
    model_manager = ModelManager(sys.argv[1], sys.argv[2])
    model_manager.test('cascade_detector/MouthDataSet/scaled/annotations.txt')
