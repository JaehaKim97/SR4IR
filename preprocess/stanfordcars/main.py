import os
from shutil import move, rmtree


if __name__ == '__main__':
    
    if not os.path.exists('datasets/archive.zip'):
        print("FAILED: The StanfordCars data path datasets/archive.zip does NOT EXIST")
    else:
        print('Unzip archive.zip ...')
        os.system('unzip -qq datasets/archive.zip -d datasets/')
            
        data_dir = os.path.join('datasets/StanfordCars')
        os.makedirs(data_dir, exist_ok=True)
        
        print('Processing the data ...')
        move('datasets/car_data/car_data/train', 'datasets/StanfordCars/')
        move('datasets/car_data/car_data/test', 'datasets/StanfordCars/')
        move('datasets/StanfordCars/test', 'datasets/StanfordCars/val')
        
        if os.path.exists('datasets/car_data'):
            rmtree('datasets/car_data')
            
        if os.path.exists('datasets/names.csv'):
            os.remove('datasets/names.csv')
            
        if os.path.exists('datasets/anno_train.csv'):
            os.remove('datasets/anno_train.csv')
            
        if os.path.exists('datasets/anno_test.csv'):
            os.remove('datasets/anno_test.csv')
            
        if os.path.exists('datasets/archive.zip'):
            os.remove('datasets/archive.zip')
        
        print('Done!')
