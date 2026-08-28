from matplotlib._mathtext import Fonts
from torch import _grid_sampler_2d_cpu_fallback
from sklearn import metrics
import matplotlib.pylab as plt
import seaborn as sns
# from ch03_funcs import metrics as ch03
from ch01_03_basics.src.ch03_funcs import metrics as ch03

def print_func(fname, result):
    """
    print function name and its result.
    Args:
        fname: function name (str)
        result: result
    Results:
        (void)
    """
    print(f"{fname} : {result}")

l1 = [0,1,1,1,0,0,0,1]
l2 = [0,1,0,1,0,1,0,0]
aaamlp_acc = ch03.accuracy(l1,l2)
aaamlp_acc2 = ch03.accuracy_v2(l1,l2)
sk_acc = metrics.accuracy_score(l1,l2)

###

print(f"accuracy_score AAAMLP: {aaamlp_acc} | AAAMLP v2: {aaamlp_acc2} | Sklearn: {sk_acc}")
print_func("true_positive",ch03.true_positive(l1,l2))
print_func("false_positive", ch03.false_positive(l1,l2))
print_func("false_negative", ch03.false_negative(l1,l2))
print_func("true_negative", ch03.true_negative(l1,l2))

###

print_func("presicion", ch03.precision(l1,l2))
print_func("recall", ch03.recall(l1,l2))
y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y_pred = [0.02638412, 0.11114267, 0.31620708,
     0.0490937, 0.0191491, 0.17554844,
     0.15952202, 0.03819563, 0.11639273,
     0.079377, 0.08584789, 0.39095342,
     0.27259048, 0.03447096, 0.04644807,
     0.03543574, 0.18521942, 0.05934905,
     0.61977213, 0.33056815]

precisions = []
recalls = []

thresholds = [0.0490937 , 0.05934905, 0.079377,
    0.08584789, 0.11114267, 0.11639273,
    0.15952202, 0.17554844, 0.18521942,
    0.27259048, 0.31620708, 0.33056815,
    0.39095342, 0.61977213]

for i in thresholds:
    temp_prediction = [1 if x >= i else 0 for x in y_pred]
    p = ch03.precision(y_true, temp_prediction)
    r = ch03.recall(y_true, temp_prediction)
    precisions.append(p)
    recalls.append(r)

plt.figure(figsize=(7,7))
plt.plot(recalls, precisions)
plt.xlabel('Recall', fontsize=15)
plt.ylabel('Precision', fontsize=15)
plt.show()

###

y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
y_pred = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0]

aaamlp_f1 = ch03.f1(y_true, y_pred)
sk_f1 = metrics.f1_score(y_true, y_pred)

print(f"f1 AAAMLP : {aaamlp_f1} | Skillearn : {sk_f1}")
###

tpr_list = []
fpr_list = []

y_true = [0, 0, 0, 0, 1, 0, 1,
0, 0, 1, 0, 1, 0, 0, 1]
# predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
0.85, 0.15, 0.99]
# handmade thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]

for thresh in thresholds:
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
    temp_tpr = ch03.tpr(y_true, temp_pred)
    temp_fpr = ch03.fpr(y_true, temp_pred)

    tpr_list.append(temp_tpr)
    fpr_list.append(temp_fpr)

plt.figure(figsize=(7,7))
plt.fill_between(fpr_list, tpr_list, alpha=0.4)
plt.plot(fpr_list,tpr_list,lw=3)
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel('TPR', fontsize=15)
plt.ylabel('FPR', fontsize=15)
plt.show()
print_func("roc_auc_score", metrics.roc_auc_score(y_true, y_pred))
# empty lists to store true positive
# and false positive values
tp_list = []
fp_list = []
# actual targets
y_true = [0, 0, 0, 0, 1, 0, 1,
0, 0, 1, 0, 1, 0, 0, 1]
# predicted probabilities of a sample being 1
y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
0.85, 0.15, 0.99]
# some handmade thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]
# loop over all thresholds
for thresh in thresholds:
    # calculate predictions for a given threshold
    temp_pred = [1 if x >= thresh else 0 for x in y_pred]
    # calculate tp
    temp_tp = ch03.true_positive(y_true, temp_pred)
    # calculate fp
    temp_fp = ch03.false_positive(y_true, temp_pred)
    # append tp and fp to lists
    tp_list.append(temp_tp)
    fp_list.append(temp_fp)

plt.figure(figsize=(7,7))
plt.fill_between(fpr_list, tpr_list, alpha=0.4)
plt.plot(fpr_list,tpr_list,lw=3)
plt.xlim(0,1.0)
plt.ylim(0,1.0)
plt.xlabel('TPR', fontsize=15)
plt.ylabel('FPR', fontsize=15)
plt.show()
y_true = [0, 0, 0, 0, 1, 0, 1,
    0, 0, 1, 0, 1, 0, 0, 1]
y_proba = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
    0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
    0.85, 0.15, 0.99]

print_func("ch03:log_loss", ch03.log_loss(y_true, y_proba))
print_func("sk:log_loss", metrics.log_loss(y_true, y_proba))
y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]
print_func("macro_precision", ch03.macro_precision(y_true, y_pred))
print_func("micro_precision", ch03.micro_precision(y_true, y_pred))
print_func("scilearn:precision_score(micro)", metrics.precision_score(y_true, y_pred, average='micro'))
print_func("weighted_precision", ch03.weighted_precision(y_true, y_pred))
print_func("scilearn:precision_score(weited)", metrics.precision_score(y_true, y_pred, average='weighted'))
print_func("weighted_f1", ch03.weighted_f1(y_true,y_pred))
print_func("Scilearn:f1_score (weighted)", metrics.f1_score(y_true,y_pred, average="weighted"))
cm = metrics.confusion_matrix(y_true,y_pred)

plt.figure(figsize=(10,10))
cmap = sns.cubehelix_palette(50,hue=0.05, rot=0, light=0.9, dark=0,as_cmap=True)

sns.set(font_scale=2.5)
sns.heatmap(cm, annot=True, cmap=cmap, cbar=False)
plt.ylabel('Actual Labels', fontsize=20)
plt.xlabel('Predicated labels', fontsize=20)
plt.show()

y_true = [
    [1, 2, 3],
    [0, 2],
    [1],
    [2, 3],
    [1, 0],
    []
  ]

y_pred = [
      [0, 1, 2],
      [1],
      [0, 2, 3],
      [2, 3, 4, 0],
      [0, 1, 2],
      [0]
    ]

for i in range(len(y_true)):
    for j in range(1, 4):
        print(
            f"""
            y_true={y_true[i]},
            y_pred={y_pred[i]},
            AP@{j}={ch03.apk(y_true[i], y_pred[i], k=j)}
            """
        )
print_func("mapk(k=1)", ch03.mapk(y_true, y_pred, k=1))
print_func("mapk(k=2)", ch03.mapk(y_true, y_pred, k=2))
print_func("mapk(k=3)", ch03.mapk(y_true, y_pred, k=3))
print_func("mapk(k=4)", ch03.mapk(y_true, y_pred, k=4))
y_true = [1, 2, 3, 1, 2, 3, 1, 2, 3]
y_pred = [2, 1, 3, 1, 2, 3, 3, 1, 2]
print_func("sklearn:cohen_kappa_score",metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic"))
print_func("sklearn:accuracy_score", metrics.accuracy_score(y_true, y_pred))
