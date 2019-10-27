from data.generate_npy_files_util import npy_files_saving_dir_path
from data.npy_file_util import load_npy_file


def main():
    x_train = load_npy_file(npy_files_saving_dir_path + 'train_x_data.npy')
    x_val = load_npy_file(npy_files_saving_dir_path + 'valid_x_data.npy')
    x_test = load_npy_file(npy_files_saving_dir_path + 'test_x_data.npy')

    y_train = load_npy_file(npy_files_saving_dir_path + 'train_labels.npy')
    y_val = load_npy_file(npy_files_saving_dir_path + 'valid_labels.npy')
    y_test = load_npy_file(npy_files_saving_dir_path + 'test_labels.npy')

    print(len(x_train) == len(y_train))
    print(len(x_val) == len(y_val))
    print(len(x_test) == len(y_test))


if __name__ == '__main__':
    main()
