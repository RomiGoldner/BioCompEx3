import sys
import random
import numpy as np
import copy

MAT_SIZE = 10
MUT_RATE = 10

def read_file(file_path):
    try:
        file = open(file_path, "r")
    except:
        return None

    number_dictionary = {num: [] for num in range(10)}
    dic_index = 0
    sample_counter = 0
    figure = ""
    for line in file.readlines():
        line = line.replace(" ", "")
        if line == "\n":
            number_dictionary[dic_index].append(figure)
            figure = ""
            sample_counter += 1
            if sample_counter == 10:
                dic_index += 1
                sample_counter = 0
        else:
            figure += line.split("\n")[0]
    return number_dictionary

def create_train_set(samples):
    dataset = []
    for value in samples.values():
        dataset.append(value[0])
    return dataset


def create_weight_matrix(dataset, train_size):
    mat_size = MAT_SIZE*MAT_SIZE
    weight_matrix = np.zeros((mat_size, mat_size))

    for i in range(0, mat_size):
        for j in range(i + 1, mat_size):
            sum = 0
            for x in dataset:
                if x[i] == x[j]:
                    sum += 1
                else:
                    sum -= 1
                weight_matrix[i, j] = sum
                weight_matrix[j, i] = weight_matrix[i, j]
    return weight_matrix, dataset


def mutate(string_number, mode):
    new_number = string_number
    for i in range(MUT_RATE):
        index = random.randint(0, len(string_number)-1)
        if string_number[index] == mode:
            new_number = new_number[:index] + '1' + new_number[index + 1:]
        else:
            new_number = new_number[:index] + mode + new_number[index + 1:]
    return new_number


def create_mutations(number_mat, picture_number, num_to_mutate, index):
    samples = []
    # for num in range(10):
    # index = random.randint(0, 9)
    number = number_mat[picture_number][index]
    for i in range(num_to_mutate):
        samples.append(mutate(number, '0'))
    return samples


def predict(weight_matrix, number_mat, mode):
    number_mat = [int(x) for x in number_mat]
    new_mat = copy.deepcopy(number_mat)
    length_mat = len(new_mat)
    iterations_count = 0
    temp_mat = copy.deepcopy(new_mat)

    # loop until convergence
    while True:
        iterations_count += 1
        list_index = list(range(length_mat))
        while list_index:
            rand_index = random.choice(list_index)
            list_index.remove(rand_index)
            mult_product = np.dot(weight_matrix[0][rand_index][:], np.array(temp_mat))

            if mult_product >= 0:
                temp_mat[rand_index] = 1
            else:
                temp_mat[rand_index] = int(mode)
        if new_mat == temp_mat:
            break
        new_mat = temp_mat
    return new_mat


def array_to_matrix(array):
    counter = 0
    number_matrix_string = ""

    list_of_lists = [array[i:i+10] for i in range(0, len(array), 10)]
    for list in list_of_lists:
        list = [str(x) for x in list]
        number_matrix_string += "".join(list)
        number_matrix_string += ('\n')

    # for val in array:
    #     if len(number_matrix_string) == 88:
    #         pass
    #     if counter < MAT_SIZE:
    #         number_matrix_string += str(val)
    #         counter += 1
    #     else:
    #         # create next line
    #         number_matrix_string += "\n"
    #         counter = 0
    return number_matrix_string

def calc_score(prediction_res, data):
    score = 0
    size = MAT_SIZE*MAT_SIZE
    for i in range(size):
        if prediction_res[i] == int(data[i]):
            score += 1
    return score/100

def calc_accuracy(prediction_res, dataset):
    min_score = 0
    min_pred_data = ""
    for data in dataset:
        score = calc_score(prediction_res, data)
        if score > min_score:
            min_score = score
            min_pred_data = data
    return min_score, min_pred_data


def main():
    file_path = sys.argv[1]
    samples = read_file(file_path)
    #TODO makr for loop to add more numbers to predict

    #TODO make two dimensionals for loop to assess the impact of mutation rate vs number to predict

    #TODO section 3 train the model on all the examples

    #TODO section 4 - change to mode to be -1 instead of 0

    #TODO section 5 - chrck error types
    zero_samples_to_predict = create_mutations(samples, 7, 10, 0)
    one_samples_to_predict = create_mutations(samples, 8, 10, 0)
    two_samples_to_predict = create_mutations(samples, 9, 10, 0)
    one_samples_to_predict.extend(zero_samples_to_predict)
    one_samples_to_predict.extend(two_samples_to_predict)
    dataset = create_train_set(samples)
    weight_matrix = create_weight_matrix(dataset, 1)

    mode = '0'


    correct_prediction_counter = 0
    for sample in one_samples_to_predict:
        new_sample_mat = predict(weight_matrix, sample, mode)
        accuracy, pred_data = calc_accuracy(new_sample_mat, dataset)
        if accuracy >= 0.9:
            correct_prediction_counter += 1
        mat = array_to_matrix(new_sample_mat)
        print("The prediction is: " + "\n" + mat)
        print("Most similar to: " + "\n" + array_to_matrix(pred_data))
    sample_size = len(one_samples_to_predict)
    print(f"correct predictions: {correct_prediction_counter} out of {sample_size}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
