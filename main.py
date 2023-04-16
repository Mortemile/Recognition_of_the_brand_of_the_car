from create_model import *

car = BrendRecognition()

car.create_sets()
car.train_model(5)

car.prepocessing_data('ferr.png')
car.test_model()
print('Правильный ответ - Ferrari')

car.prepocessing_data('mercedes.png')
car.test_model()
print('Правильный ответ - Mercedes')

car.prepocessing_data('reno.png')
car.test_model()
print('Правильный ответ - Renault')
