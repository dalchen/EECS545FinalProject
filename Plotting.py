from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#This function takes in training and test data, calculates the logistic regression function, predicts test data, and returns the error
def calculateError(x_train, y_train, x_test, y_test, lambda_value):
    clf = LogisticRegression(random_state=0, solver='lbfgs', C=1/lambda_value, multi_class='multinomial').fit(x_train, y_train)
    y_test_predict = clf.predict(x_test)
    error = 1 - accuracy_score(y_test_predict, y_test)
    return error

###Sampled data points for each strategy
###training data set will have total data equal to 'num_total_sampled_data_points'
random_train_x = 0
random_train_y = 0
random_test_x = 0
random_test_y = 0
margin_train_x = 0
margin_train_y = 0
margin_test_x = 0
margin_test_y = 0

#Initialize parameters and total number of labeled points
num_total_sampled_data_points = 100
lambda_value = 10**(-4)#This needs to be tuned

#Initialize vectors to be used for plotting
error_random_vector = []
error_margin_vector = []
num_samples_vector = []

#Iterating through number of samples, and adding the resulting errors to plotting vectors
for num_samples in range(3, num_total_sampled_data_points):#Each iteration you are using more labeled data points to train
    num_samples_vector.append(num_samples)
    error_random_vector.append(calculateError(random_train_x[:num_samples,:],random_train_y[:num_samples],random_test_x,random_test_y,lambda_value))
    error_margin_vector.append(calculateError(margin_train_x[:num_samples,:],margin_train_y[:num_samples],margin_test_x,margin_test_y,lambda_value))

#Plotting
plt.gca().set_color_cycle(['red', 'green'])
plt.plot(num_samples_vector, error_output_random)
plt.plot(num_samples_vector, error_output_margin)
plt.legend(['Random', 'Margin'], loc='upper right')
plt.xlabel("Number Of Labels")
plt.ylabel("Error")
plt.show()
