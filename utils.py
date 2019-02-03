import numpy as np


def resnet_split(m):
    return (m[0][6], m[1])

def stratified_split_indexes(df, valid_pct, seed):
    valid_images = set()
    for id, id_group in df.groupby('Id'):
        id_images = sorted(id_group.Image.tolist())
        if len(id_images) > 1:
            np.random.seed(seed)
            valid_images.update(np.random.choice(id_images, max(1, int(valid_pct * len(id_images)))))

    valid_bitmap = df.Image.map(lambda image: image in valid_images)
    return np.arange(len(df))[valid_bitmap]

def filter_samples_for_classes_with_few_occurrence(df, min_occurrences_to_include):
    id_group_df = df.groupby('Id').agg({'Image': 'count'}).reset_index()
    few_occurrence_ids = id_group_df[id_group_df.Image < min_occurrences_to_include].Id.unique()
    filter_bitmap = df.Id.map(lambda id: id not in few_occurrence_ids)
    return df[filter_bitmap]

