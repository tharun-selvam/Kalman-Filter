import numpy
import numpy as np
import matplotlib.pyplot as plt

data_text = open('kalmann.txt', 'r')

def removeComma(list):
    for i in list:
        if (i == ','):
            list.remove(i)

    return list


def makeListFloat(list):
    for i in range(len(list)):
        list[i] = float(list[i])

    return list


def getRestOfTheData(data_text):
    # initial pos is a numpy array of type float and shape (1, 2)
    initial_pos = data_text.readline()
    initial_pos = initial_pos.split()
    initial_pos = removeComma(initial_pos)
    initial_pos = makeListFloat(initial_pos)
    initial_pos = np.array(initial_pos)
    initial_pos = initial_pos.reshape((1, 2))

    # getting the remaining lists as a numpy array of type float and shape (n, 4)
    state_stacked = data_text.readline()
    state_stacked = state_stacked.split()
    state_stacked = makeListFloat(removeComma(state_stacked))
    state_stacked = np.array(state_stacked)
    state_stacked = numpy.reshape(state_stacked, (1, 4))

    while True:

        state = data_text.readline()

        if not state:
            break

        state = state.split()
        state = makeListFloat(removeComma(state))
        state = np.array(state)
        state = numpy.reshape(state, (1, 4))

        state_stacked = numpy.concatenate((state_stacked, state), axis=0)

    return initial_pos, state_stacked


def plotPoints(entire_data, initial_pos, predicted_data=None):
    '''
        enitre_data: It contains all the points in state_stacked
    '''

    plt.plot(initial_pos[0, 0], initial_pos[0, 1], '*')
    plt.plot(entire_data[:, 0], entire_data[:, 1], color='r')
    plt.plot(entire_data[-1, 0], entire_data[-1, 1], 'o')
    plt.plot(predicted_data[:, 0], predicted_data[:, 1], color='g')
    plt.plot(predicted_data[-1, 0], predicted_data[-1, 1], 'o')
    plt.show()

initial_pos, state_stacked = getRestOfTheData(data_text)
# plotPoints(state_stacked, initial_pos)

# Running a few trail runs
measured1 = state_stacked[0]
X_init = numpy.reshape(initial_pos, (2, 1))
M_init = numpy.reshape(measured1[:2], (2, 1))

r1, r2 = 10000, 10000
P_init = np.array([[r1, 0],
                   [0, r2]], dtype='float')
velocity_init = numpy.reshape(measured1[2:], (2, 1))
mat_A = np.array([[1, 1],
                  [0, 1]])

a, b = 5, 5
Q = np.array([[a, 0],
              [0, b]], dtype='float')

X_pred = X_init + velocity_init # assuming delta T is 1
P_pred = mat_A @ P_init @ mat_A.T + Q

c, d = .1, .1
R = np.array([[c, 0],
              [0, d]], dtype='float')

kalmanGain = P_pred/(P_pred+R)

kalmanGain[0, 1], kalmanGain[1, 0] = 0, 0
P_pred[0, 1], P_pred[1, 0] = 1, 1

X_actual = X_pred + kalmanGain @ (M_init - X_pred)
P_actual = (np.ones_like(kalmanGain) - kalmanGain) @ P_pred

P_actual[0, 1], P_actual[1, 0] = 0, 0

print(X_actual, '\n', M_init, '\n')
X_actual_stacked = np.reshape(X_actual, (1, 2))


loss = []

for i in range(1, len(state_stacked)):
    measured1 = state_stacked[i]
    X_init = numpy.reshape(X_actual, (2, 1))
    M_init = numpy.reshape(measured1[:2], (2, 1))

    # r1, r2 = 10000, 10000
    P_init = P_actual
    velocity_init = numpy.reshape(measured1[2:], (2, 1))
    mat_A = np.array([[1, 1],
                      [0, 1]])

    a, b = 5, 5
    Q = np.array([[a, 0],
                  [0, b]], dtype='float')
    c, d = .00001, .00001
    R = np.array([[c, 0],
                  [0, d]], dtype='float')

    X_pred = X_init + velocity_init # assuming delta T is 1
    P_pred = mat_A @ P_init @ mat_A.T + Q

    kalmanGain = P_pred/(P_pred+R)

    kalmanGain[0, 1], kalmanGain[1, 0] = 0, 0
    P_pred[0, 1], P_pred[1, 0] = 0, 0

    X_actual = X_pred + kalmanGain @ (M_init - X_pred)
    P_actual = (np.ones_like(kalmanGain) - kalmanGain) @ P_pred

    # print(X_actual, '\n', M_init, '\n')
    X_actual = np.reshape(X_actual, (1, 2))
    X_actual_stacked = numpy.concatenate((X_actual_stacked, X_actual), axis=0)

    P_actual[0, 1], P_actual[1, 0] = 1, 1


loss = state_stacked[:, :2] - X_actual_stacked
print(loss)

plotPoints(state_stacked, initial_pos, X_actual_stacked)

print(c, d)
