## The Classiffier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_vOQ0QBDepiRg7pM2YDrIGL0hUgyARdJ) 


SHAP model to explaine the model output.
We used SHAP to explain and compute the contribution of each feature in our model.

```python
def SHAP(model, k_features):
        model.split_corpus_basic()
        y = model.train_labels 
        y.replace(['removed', 'shadow_ban'], 'not_exist', inplace=True)
        model.df_train.drop(columns=['created_date'], inplace=True)
        X = model.df_train*1
        
        #train XGBoost model
        model1 = xgboost.XGBClassifier(max_depth=5, learning_rate=0.5).fit(X, y)

        #compute SHAP values
        explainer = shap.Explainer(model1, X, link=shap.links.logit)
        shap_values = explainer(X)        
        shap.plots.bar(shap_values)
        
        vals = np.abs(shap_values.values).mean(0)
        feature_names = X.columns

        feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        k_featues_name = [f_name for f_name, val in feature_importance.head(k_features).values.tolist()]
        return k_featues_name
```

![alt text](https://i.postimg.cc/fTbKCxBG/Whats-App-Image-2022-06-27-at-17-56-28.jpg)
