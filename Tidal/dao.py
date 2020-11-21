import csv
def score(y_pred):
    y_true = []
    with open('data/label_4_refer.csv', 'r') as file:
        reader = csv.reader(file)
        for i in reader:
            if (i[0] == '1'):
                y_true.append([0, 1])
            else:
                y_true.append([1, 0])
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    index = 0
    for i in y_true:
        if (i[1] == 1):
            if (y_pred[index] == 1):
                tp += 1
            else:
                fn += 1
        else:
            if (y_pred[index] == 1):
                fp += 1
            else:
                tn += 1
        index += 1
    print('tp ', tp, ' fn ', fn, ' fp ', fp, ' tn ', tn)
    print('accuracy: ', (tp + tn) / (tp + tn + fp + fn))
    print('precision: ', tp / (tp + fp))
    print('recall: ', tp / (tp + fn))
    print('f1: ', 2 * tp / (2 * tp + fp + fn))