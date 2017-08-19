from sklearn import svm
import numpy as np
import csv

def create_model(training_data, training_result):
  clf = svm.SVC()
  return clf.fit(training_data, training_result)

def predict_output(model, test_data):
  predicted = model.predict(test_data)
  return predicted

def fetch_training_data():
  file_path = '../data/train.csv'
  file = open(file_path, "r")
  reader = csv.reader(file)
  training_array = []
  training_result_array = []
  index = 0
  for line in reader:
    index += 1
    if index == 1:
      continue
    record_array = []
    record_array.append(int(line[2]) if line[2] != '' else 4)
    record_array.append(0 if line[4] == 'male' else 1)
    record_array.append(float(line[5]) if line[5] != '' else -1)
    record_array.append(int(line[6]) if line[6] != '' else 0)
    record_array.append(int(line[7]) if line[7] != '' else 0)
    record_array.append(float(line[9]) if line[9] != '' else -1)
    if line[11] == 'C':
      record_array.append(0)
    elif line[11] == 'Q':
      record_array.append(1)
    else:
      record_array.append(2)
    training_array.append(record_array)
    training_result_array.append(line[1])
  return training_array, training_result_array

def fetch_test_data():
  file_path = '../data/test.csv'
  file = open(file_path, "r")
  reader = csv.reader(file)
  test_array = []
  passenger_ids = []
  index = 0
  for line in reader:
    index += 1
    if index == 1:
      continue
    passenger_ids.append(line[0])
    record_array = []
    record_array.append(int(line[1]) if line[1] != '' else 4)
    record_array.append(0 if line[3] == 'male' else 1)
    record_array.append(float(line[4]) if line[4] != '' else -1)
    record_array.append(int(line[5]) if line[5] != '' else 0)
    record_array.append(int(line[6]) if line [6] != '' else 0)
    record_array.append(float(line[8]) if line[9] != '' else -1)
    if line[10] == 'C':
      record_array.append(0)
    elif line[10] == 'Q':
      record_array.append(1)
    else:
      record_array.append(2)
    test_array.append(record_array)
  return passenger_ids, test_array

def write_output_in_csv(passenger_ids, predicted_output):
  file = open('../data/output.csv', "w")
  file.write("PassengerId,Survived")
  file.write("\n")
  for index in range(0, len(passenger_ids)):
    print(passenger_ids[index]+','+predicted_output[index])
    file.write(passenger_ids[index]+','+predicted_output[index])
    file.write("\n")
  file.close()


training_data, training_result = fetch_training_data()
passenger_ids, test_array = fetch_test_data()
model = create_model(training_data, training_result)
predicted_output = predict_output(model, test_array)
write_output_in_csv(passenger_ids, predicted_output)

