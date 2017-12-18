import pandas as pd
import graphlab as gl
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    # Load training ratings for users and jokes
    sf = gl.SFrame('data/ratings.dat', format='tsv')

    # Load sample submission to test predictions against
    df_sample = pd.read_csv('data/sample_submission.csv')
    sf_sample = gl.SFrame(df_sample)

    return sf, sf_sample, df_sample

def score(df_true, df_predict):
    """Look at 5% of most highly predicted jokes for each user.
    Return the average actual rating of those jokes.
    """
    #sample = pd.read_csv('data/sample_submission.csv')

    df = pd.concat([#sample,
                    df_predict,
                    df_true], axis=1)


    g = df.groupby('user_id')

    top_5 = g.pred_rating.apply(
        lambda x: x >= x.quantile(.95)
    )

    return df_true[top_5==1].mean()['true_rating']

def create_factorization_recommender(sf, num_factors=32):
    m = gl.recommender.item_similarity_recommender.create(observation_data=sf,
                                                        user_id='user_id', item_id='joke_id',
                                                        target='rating',
                                                        similarity_type='pearson',
                                                        num_factors=num_factors)
    return m

if __name__ == "__main__":
    sf, sf_sample, df_sample = load_data()

    train_data, val_data = gl.recommender.util.random_split_by_user(sf, 'user_id', 'joke_id')

    df_true= pd.DataFrame()
    df_predict = pd.DataFrame()

    df_true['user_id'] = val_data['user_id']
    df_true['joke_id'] = val_data['joke_id']
    df_true['true_rating'] = val_data['rating']

    # Plot scores vs. num_factors

    # hyperparameters = [num_factors, similarity_type_list, thresholds, only_top_k_list]

    # threshold = list(np.linspace(2,64,num=62))

    # similarity_type = ['pearson', 'cosine', 'jaccard']
    num_factors = range(71)
    scores=[]

    for i in num_factors:
        m = create_factorization_recommender(train_data, num_factors= i)

        df_predict['pred_rating'] = m.predict(val_data)
        rc= score(df_true, df_predict)
        scores.append(rc)
        print(num_factors,': ', i, 'Score: ', rc)

    plt.plot(num_factors, scores)
    plt.title('Mean Score vs Number of Factors')
    plt.xlabel("Number of Factors")
    plt.ylabel('Score')
    plt.show()

    # sample_sub_fname = "data/sample_submission.csv"
    # ratings_data_fname = "data/ratings.dat"
    # output_fname = "data/test_ratings.csv"

    # ratings = gl.SFrame(ratings_data_fname, format='tsv')
    # sample_sub = pd.read_csv(sample_sub_fname)
    # for_prediction = gl.SFrame(sample_sub)
    # rec_engine = gl.item_similarity_recommender.create(observation_data=ratings,
    #                                                  user_id="user_id",
    #                                                  item_id="joke_id",
    #                                                  target='rating')
    #
    # sample_sub.rating = rec_engine.predict(for_prediction)
    # sample_sub.to_csv(output_fname, index=False)
