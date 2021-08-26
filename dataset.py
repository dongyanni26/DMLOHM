import numpy as np
from torch.utils.data import Dataset

class TripletTrainData(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, HSIdataset):
        self.HSIdataset = HSIdataset
        train_data = []
        train_labels = []

        for step, (i, j) in enumerate(self.HSIdataset):
            train_data_temp = [i]
            train_data += train_data_temp
            train_labels_temp = [j]
            train_labels += train_labels_temp 
        
        self.train_labels = np.array(train_labels)
        self.train_data = train_data
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                    for label in self.labels_set}

    def __getitem__(self, index):
        img1, label1 = self.train_data[index], self.train_labels[index].item()
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])
        negative_label = np.random.choice(list(self.labels_set - set([label1])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        img2 = self.train_data[positive_index]
        img3 = self.train_data[negative_index]

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.HSIdataset)

class TripletTestData(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, HSIdataset):
        self.HSIdataset = HSIdataset

        self.test_labels = self.HSIdataset.test_labels
        self.test_data = self.HSIdataset.test_data
        # generate fixed triplets for testing
        self.labels_set = set(self.test_labels.numpy())
        self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                    for label in self.labels_set}

        random_state = np.random.RandomState(29)

        triplets = [[i,
                        random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                        random_state.choice(self.label_to_indices[
                                                np.random.choice(
                                                    list(self.labels_set - set([self.test_labels[i].item()]))
                                                )
                                            ])
                        ]
                        for i in range(len(self.test_data))]
        self.test_triplets = triplets

    def __getitem__(self, index):
        img1 = self.test_data[self.test_triplets[index][0]]
        img2 = self.test_data[self.test_triplets[index][1]]
        img3 = self.test_data[self.test_triplets[index][2]]

        return (img1, img2, img3), []

    def __len__(self):
        return len(self.HSIdataset)