from sklearn import metrics


def intrinsic_metrics(data, predictions):
    #sillhouette score ranges from -1 to 1, where 1 is best and 0 indicates cluster overlap
    ss = metrics.silhouette_score(data, predictions, metric='euclidean')
    print("Sillhouette score:", ss)
    # variance ratio criterion-- how tightly clustered (higher is better)
    chs = metrics.calinski_harabasz_score(data, predictions)
    print("Calinski-Harabasz Index:", chs)
    # similarity between clusters (lower is better)
    dbs = metrics.davies_bouldin_score(data, predictions)   
    print("Davies-Bouldin Index:", dbs)
    return [ss, chs, dbs]


def extrinsic_metrics(ground_truths, predictions):
    # rand index score (higher is better, max 1)
    rs = metrics.rand_score(ground_truths, predictions)
    print("Random Index:", rs)
    # homogeneity - closer to 1 is better
    homo = metrics.homogeneity_score(ground_truths, predictions)
    print("Homogeneity:", homo)
    
    return [rs, homo]
