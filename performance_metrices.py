actual=       [1, 1, 0, 1, 0] # actual
predicted =   [0, 1, 0, 1, 0] 
probailities = []


def confusion_matrix_2classes(actual, predicted, thershold):
    unique = sorted(set(actual))
    # imap   = {className: index for index, className in enumerate(unique)}
    matrix = [[0 for _ in unique] for _ in unique] #empty matrix N*N
    for i in range(len(predicted)):
        if actual[i] == 0:
                if predicted[i] < thershold:
                        matrix[0][0] = matrix[0][0] + 1
                else:
                        matrix[0][1] = matrix[0][1] + 1
        elif actual[i] == 1:
                if predicted[i] >= thershold:
                        matrix[1][1] = matrix[1][1] +1
                else:
                        matrix[1][0] = matrix[1][0] +1

    true_positive  = matrix[1][1] 
    false_negative = matrix[1][0]
    false_positive = matrix[0][1]
    true_negative  = matrix[0][0]
    return matrix


# for p, a in zip(predicted, actual):

#     matrix[imap[p]][imap[a]] += 1
def recall(matrix):
    """Given a className in the confusion matrix, return the recall
    that corresponds to this className. The recall is defined as:

    - recall = true positive / (true positive + false positive)

    :param className: className used in the ConfusionMatrix
    :return: the recall corresponding to ``className``.
    :rtype: float
    """
    true_positive  = matrix[1][1] 
    false_negative = matrix[1][0]

    if false_negative == 0:
        return 0.0
    return true_positive / (true_positive + false_negative )   

def precision(matrix):
    """Given a className in the confusion matrix, return the precision
    that corresponds to this className. The precision is defined as:

    - precision = true positive / (true positive + false negative)

    :param className: className used in the ConfusionMatrix
    :return: the precision corresponding to ``className``.
    :rtype: float
    """
    true_positive  = matrix[1][1] 
    false_positive = matrix[0][1]
    if false_positive == 0:
        return 0.0
    return true_positive / false_positive

def accuracy(matrix):
    """
    Given a className in the confusion matrix, return the accuracy
    that corresponds to this className. The accuracy is defined as:

    - precision = (true positive + true negative)/(false negative + false positive)

    :param className: className used in the ConfusionMatrix
    :return: the accuracy corresponding to ``className``.
    :rtype: float
    """
    true_positive  = matrix[1][1] 
    false_negative = matrix[1][0]
    false_positive = matrix[0][1]
    true_negative  = matrix[0][0]
    acc = (true_positive + true_negative)/(false_negative + false_positive)
    return acc

# def ROC ()
print('confusion matrix: ',confusion_matrix_2classes(actual, predicted, 0.5))
print(recall(confusion_matrix_2classes(actual, predicted, 0.5)))

# def compare(wights, percent):
    
