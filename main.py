import sys
from models.ModelManager import ModelManager

if __name__ == '__main__':
    model_manager = ModelManager(sys.argv[1])
    model_manager.train()
