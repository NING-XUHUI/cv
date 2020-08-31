from sklearn import linear_model

X = [[20, 3],
     [23, 7],
     [31, 10],
     [42, 13],
     [50, 7],
     [60, 5]]

Y = [0,
     1,
     1,
     1,
     0,
     0]

lr = linear_model.LogisticRegression()
lr.fit(X, Y)

testX = [[28, 8]]

label = lr.predict(testX)
print("predicted label = ", label)

prob = lr.predict_proba(testX)
print("probability = ", prob)
