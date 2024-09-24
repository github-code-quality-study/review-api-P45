import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        global reviews

        if environ["REQUEST_METHOD"] == "GET":

            # Write your code here

            query_string = environ.get("QUERY_STRING")
            parameters = parse_qs(query_string)

            location = parameters.get("location", [None])[0]
            start_date = parameters.get("start_date", [None])[0]
            end_date = parameters.get("end_date", [None])[0]

            valid_locations =["Albuquerque, New Mexico","Carlsbad, California","Chula Vista, California","Colorado Springs, Colorado","Denver, Colorado","El Cajon, California","El Paso, Texas","Escondido, California","Fresno, California","La Mesa, California","Las Vegas, Nevada","Los Angeles, California","Mesa, Arizona","Oceanside, California","Phoenix, Arizona","Sacramento, California","Salt Lake City, Utah","San Diego, California","Tucson, Arizona"]

            
            filtered_reviews = [review for review in reviews]   

            # Filter reviews by sentiment
            filtered_sentiments =[]

            for review in filtered_reviews:
                sentiment_score = self.analyze_sentiment(review['ReviewBody'])
                filtered_sentiments.append({
                    "ReviewId": review.get("ReviewId", str(uuid.uuid4())),
                    "ReviewBody": review["ReviewBody"],
                    "Location": review["Location"],
                    "Timestamp": review["Timestamp"],
                    "sentiment": sentiment_score
                })
            filtered_sentiments.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            filtered_reviews = filtered_sentiments

            # Filter reviews by location
            if location:
                if location not in valid_locations:
                    response_body = json.dumps({"error": "Invalid location"}).encode("utf-8")
                    start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                    ])
                    return [response_body]

                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

            #Filter reviews by date
            if start_date or end_date:
                try:
                    if start_date :
                        start_date = datetime.strptime(start_date, '%Y-%m-%d')
                    if end_date :
                        end_date = datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    response_body = json.dumps({"error": "Invalid date format"}).encode("utf-8")
                    start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                    ])
                    return [response_body]
                def within_date_range(review_date_range):
                    review_date = datetime.strptime(review_date_range.split( )[0], '%Y-%m-%d')
                    if start_date and end_date:
                        return start_date <= review_date <= end_date
                    if start_date:
                        return review_date >= start_date
                    if end_date:
                        return review_date <= end_date
                    
                filtered_reviews = [review for review in filtered_reviews if within_date_range(review['Timestamp'])]
            

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]

        

        if environ["REQUEST_METHOD"] == "POST":

            # Read the content length from the request headers
            content_length = int(environ.get("CONTENT_LENGTH", 0))
            # Read the request body data
            request_body = environ["wsgi.input"].read(content_length).decode("utf-8")
            # Parse the request body data
            request_data = parse_qs(request_body)


            valid_locations =["Albuquerque, New Mexico","Carlsbad, California","Chula Vista, California","Colorado Springs, Colorado","Denver, Colorado","El Cajon, California","El Paso, Texas","Escondido, California","Fresno, California","La Mesa, California","Las Vegas, Nevada","Los Angeles, California","Mesa, Arizona","Oceanside, California","Phoenix, Arizona","Sacramento, California","Salt Lake City, Utah","San Diego, California","Tucson, Arizona"]

            review_body = request_data.get("ReviewBody",[None])[0]
            review_location = request_data.get("Location",[None])[0]

            if review_body is None or review_location is None:
                response_body = json.dumps({"error": "Missing required fields"}).encode("utf-8")
                start_response("400 Bad Request", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
            
            if review_location not in valid_locations:
                response_body = json.dumps({"error": "Invalid location"}).encode("utf-8")
                start_response("400 Bad Request", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
            
            new_review = {
                "ReviewId": str(uuid.uuid4()),
                "ReviewBody": review_body,
                "Location": review_location,
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            reviews.append(new_review)
            df_reviews = pd.DataFrame(reviews)
            df_reviews.to_csv('data/reviews.csv', index=False)

            # Create the response body from the sentiment scores and convert to a JSON byte string
            response_body = json.dumps(new_review, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]
        

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()