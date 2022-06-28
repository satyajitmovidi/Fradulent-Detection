input_data=(1,2,10000,5000,15000,0,10)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)


print(prediction)

print(training_data_accuracy,test_data_accuracy)
