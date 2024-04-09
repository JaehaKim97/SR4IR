import os

if __name__ == '__main__':
    os.makedirs('experiments', exist_ok=True)
    os.makedirs('tb_loggers', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)

    print("Setup Done:)")
