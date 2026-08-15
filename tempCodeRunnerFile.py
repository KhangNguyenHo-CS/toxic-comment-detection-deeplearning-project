
# for batch in test.as_numpy_iterator():
#     #unpack the batch
#     x_true, y_true = batch

#     #Make a prediction
#     yhat = model.predict(x_true)


#     #Flatten the predictions
#     y_true = y_true.flatten()
#     yhat = yhat.flatten()

#     pre.update_state(y_true, yhat)
#     re.update_state(y_true, yhat)
#     acc.update_state(y_true, yhat)
