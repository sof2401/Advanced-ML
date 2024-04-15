# Writing a demo custom function for .py file
def demo_function(name):
    print(f'Hello, {name}!')

def regression_metrics(y_true, y_pred, label='', verbose=True,
                       output_dict=False):

  mae=mean_absolute_error(y_true, y_pred)
  mse= mean_squared_error(y_true, y_pred)
  rmse= mean_squared_error(y_true, y_pred, squared=False)
  r_squared= r2_score(y_true, y_pred)
  if verbose == True:

    header='-'*60
    print (header, f"Regression Metrics: {label}", header, sep='\n')
    print(f'-MAE = {mae:.3f}')
    print(f'-MSE={mse:,.3f}')
    print(f'-RMSE={rmse:,.3f}')
    print (f'-R^2={r_squared:,.3f}')
  if output_dict == True:
    metrics= {'Label':label, 'MAE':mae,
             'MSE':mse, 'RMSE':rmse, 'R^2':r_squared}
    return metrics

def evaluate_regression(reg, X_train, y_train,X_test, y_test, verbose= True,
                        output_frame=False):
  y_train_pred= reg.predict(X_train)

  results_train= regression_metrics(y_train, y_train_pred, verbose= verbose,
                                    output_dict= output_frame, label='Training Data')
  print()

  y_test_pred= reg.predict(X_test)

  results_test= regression_metrics(y_test, y_test_pred, verbose=verbose,
                                   output_dict=output_frame, label='Test Data')

  if output_frame:
    results_df= pd.DataFrame([results_train, results_test])

    results_df=results_df.set_index('Label')

    results_df.index.name= None

    return results_df.round(3)