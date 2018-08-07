import pandas as pd
import scipy.sparse as sparse
import implicit
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv("data.csv")

df['freq'] = df.groupby(['user','item'])['user'].transform('count')

df.drop_duplicates(subset=['user','item'], keep='first', inplace = True)

df = df[['user','item','freq']]

# Create a numeric user_id and item_id feature
df['user'] = df['user'].astype("category")
df['item'] = df['item'].astype("category")
df['user_id'] = df['user'].cat.codes
df['item_id'] = df['item'].cat.codes

# The implicit library expects data as a item-user matrix so we need to create two matrices
# one for fitting the model (item-user) & another for recommendations (user-item)
sparse_item_user = sparse.csr_matrix((df['freq'].astype(float), (df['item_id'], df['user_id'])))
sparse_user_item = sparse.csr_matrix((df['freq'].astype(float), (df['user_id'], df['item_id'])))

# ALS model
model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

# Calculate the confidence by multiplying it by our alpha value.
alpha_val = 15
data_conf = (sparse_item_user * alpha_val).astype('double')

#Fit the model
model.fit(data_conf)
    
#****************************
# CREATE USER RECOMMENDATIONS
#****************************
while True:
    print()
    user_inp = int(input('Enter User# for Recommendation: '))
    x = df['user_id'][df.user == user_inp]
    user_id = x.iloc[0]
    # Use the implicit recommender.
    recommended = model.recommend(user_id, sparse_user_item)
    items = []
    scores = []
    for item in recommended:
        idx, score = item
        items.append(df.item.loc[df.item_id == idx].iloc[0])
        scores.append(score)
    recommendations = pd.DataFrame({'item': items, 'score': scores})
    print(recommendations)
    print()
    flag = input('Need Recommendation for another User? - Enter Y/N: ')
    if (flag != 'Y' and flag != 'y'):
        print('Goodbye...!!!')
        break