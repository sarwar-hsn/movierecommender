from flask import Flask, request, jsonify
from flask_restful import Resource, Api

# loading trained model when the server starts
from recommender.recommend import *

app = Flask(__name__)
api = Api(app)

#movie recommend api end point
class MovieRecommend(Resource):
    def post(self):
        movie_name = request.args.get('movie_name')  # Getting the movie name from the query parameter
        try:
            movies = get_recommendations(movie_name)
        except:
            return jsonify({"error": "something went wrong!"})

        response = []
        for index, row in movies.iterrows():
            response.append({'id': row['id'], 'title': row['original_title']}) #adding id and movie title
        return jsonify(response)

api.add_resource(MovieRecommend, '/recommend')


#api end point for getting movie info by id
class MovieInfo(Resource):
    def get(self, movie_id):
        print(movie_id)
        movie = get_movie_by_id(movie_id)
        if movie:
            return jsonify({"body": movie})
        else:
            return jsonify({"error": "Movie not found"})

api.add_resource(MovieInfo, '/info/<int:movie_id>')


if __name__ == '__main__':
    app.run(debug=True)
