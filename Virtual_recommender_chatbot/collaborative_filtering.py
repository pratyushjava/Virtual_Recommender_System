import sys
import math
import gzip
import time
import numpy

#data_path = "../../data/user_dedup.json.gz"
data_path = "../../data/reviews_200k.json.gz"

class NestedDict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value
    
    
def data_parser(path):
    g = gzip.open(path, 'r')
    for line in g:
        yield eval(line)
        
def adjusted_cosine_similarity(user_ratings,neighbor_rating):
    common_ratings = 0
    sum_ij = 0
    sum_square_i = 0
    sum_square_j = 0
    mean_i = numpy.mean(list(user_ratings.values())) 
    mean_j = numpy.mean(list(neighbor_rating.values())) 
    for pid in user_ratings:
        i = user_ratings[pid] 
        if pid in neighbor_rating:
            j = neighbor_rating[pid]
            common_ratings += 1
            sum_ij += (i-mean_i)*(j-mean_j)
            sum_square_i += pow(i-mean_i, 2)
            sum_square_j += pow(j-mean_j, 2)
        else:
            j = 0
            sum_square_i += pow(i-mean_i, 2)
            
    if (len(neighbor_rating) > common_ratings):
        for pid in neighbor_rating:
            j = neighbor_rating[pid] 
            if pid not in user_ratings:
                i = 0
                sum_square_j += pow(j-mean_j, 2)
                
    root_product = math.sqrt(sum_square_i) * math.sqrt(sum_square_j)
    if(root_product == 0):
        return 0
    
    return sum_ij / root_product


def get_neighbor(user_id, data_nestdict):
    
    user_distance_tuples = []
    for neighbor_ID in data_nestdict:
        if neighbor_ID != user_id:
            distance = adjusted_cosine_similarity(data_nestdict[user_id], data_nestdict[neighbor_ID])
            user_distance_tuples.append((neighbor_ID, distance))
            
    user_distance_tuples.sort(key=lambda itemTuple: itemTuple[1], reverse=True)
    return user_distance_tuples
    


def get_recommendations(user_id, data_nestdict, k, n):
    neighbors_distance = 0.0
    recommendations = {}
    knn = get_neighbor(user_id, data_nestdict)
    test_user_ratings = data_nestdict[user_id]
    for i in range(k):
        neighbors_distance += knn[i][1]
        
    if(neighbors_distance > 0):
        for i in range(k):
            neighbor_importance = knn[i][1] / neighbors_distance
            neighbor_ID = knn[i][0]
            neighbor_ratings = data_nestdict[neighbor_ID]
            for pid in neighbor_ratings:
                if pid not in test_user_ratings:
                    if pid not in recommendations:
                        recommendations[pid] = neighbor_ratings[pid] * neighbor_importance
                    else:
                        recommendations[pid] += neighbor_ratings[pid] * neighbor_importance
                        
    recommendations = list(recommendations.items())
    recommendations.sort(key=lambda itemTuple: itemTuple[1], reverse=True)

    return recommendations[:n]
        
        
def show_recommendations(user_id, data_nestdict, k, n):
    
    recommendations = get_recommendations(user_id, data_nestdict, k, n)

    print("following are the recommendations for user = ", user_id)
    print("total number of recommendations = ",len(recommendations))
    for recommendation in recommendations:
        print(str(recommendation))

    

    
    
    
def main():
    if len(sys.argv) != 4:
        print("Please provide [user id] [k] [n] as command line argument")
        print("where k = number of nearest neighbor")
        print("where n = number of recommendation")
        sys.exit(1)
        
    user_id = sys.argv[1] 
    k = int(sys.argv[2]) 
    n = int(sys.argv[3]) 
    data_nestdict = NestedDict()
    
    for ratings in data_parser(data_path):
        uID = ratings['reviewerID']
        pID = ratings['asin']
        rate = ratings['overall']
        data_nestdict[uID][pID] = rate
        
    show_recommendations(user_id, data_nestdict, k, n)
    
main()
        