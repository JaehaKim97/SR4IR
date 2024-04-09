import os


if __name__ == '__main__':
    
    if not os.path.exists('datasets/VOCtrainval_11-May-2012.tar'):
        print("FAILED: The data path datasets/VOCtrainval_11-May-2012.tar does NOT EXIST")
    else:
        # make directories
        os.makedirs('datasets/VOC', exist_ok=True)

        # untar VOC
        print('Untar VOCtrainval_11-May-2012.tar ...')
        os.system('tar -xf datasets/VOCtrainval_11-May-2012.tar -C datasets/VOC')

        # remove .tar file
        if os.path.exists('datasets/VOCtrainval_11-May-2012.tar'):
            os.remove('datasets/VOCtrainval_11-May-2012.tar')

        print('Done!')
