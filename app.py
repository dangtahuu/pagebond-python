from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# import joblib  # Assuming you meant joblib, but you can replace it with your pickle library

app = Flask(__name__)

model = pickle.load(open('data/model.pkl','rb'))
df = pickle.load(open('data/data.pkl','rb'))

def recommend_books(id):
    similar_books = df[['title', 'author','_id']].copy()
    index_list = similar_books.index[similar_books['_id'] == id]

    similarity_scores = cosine_similarity(model[index_list[0]], model).flatten()
    similar_books['similarity'] = similarity_scores
    similar_books = similar_books.sort_values(by='similarity', ascending=False)
    
    result = similar_books.head(30)
    id_array = result['_id'].values.tolist()
    return id_array


@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get book_id from the query parameters
        book_id = request.args.get('book_id')

        if not book_id:
            raise ValueError('book_id parameter is missing.')

        # Make predictions
        result = recommend_books(book_id)

        # Return the result as JSON
        return jsonify({'result': result})
        # return result
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
